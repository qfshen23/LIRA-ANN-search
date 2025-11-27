'''
LIRA for small-scale datasets
1. Redundancy can be set through repa_step
2. Internal index uses FLAT for exact search within each partition.
cluster, bucket, partition are used interchangeably.
'''
import numpy as np
import os
num_threads = 32
os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
import time
from utils import *
import pandas as pd
from dataclasses import dataclass
from transformers import HfArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np
from prettytable import PrettyTable
from model_probing import *
from tqdm import tqdm
import time

@dataclass
class Config:
    method_name: str = 'LIRA_RE'
    dataset: str = None # dataset name, e.g., 'sift', 'spacev10m', 'tiny5m', 'bigann10m' (required)
    data_path: str = '/data/vector_datasets' # base path for datasets
    dis_metric: str = 'L2'
    k: int = None # number of nearest neighbors (required)
    n_bkt: int = None # number of buckets/clusters (required)
    n_epoch: int = 10 # model training epoch, 10 for small-scale; 30 for large-scale
    batch_size: int = 64
    n_mul: int = 2

    repa_step: int = 10 # repartition step, redundancy 1/repa_step data in each step (deprecated when using redundancy_ratio)
    redundancy_ratio: float = 0.03 # 冗余比例，只冗余前 x% 的 base vector (例如 0.1 表示 10%)
    duplicate_type: str = 'model' # 'None' 'model'

    pth_log: str = None  # 将在 update() 中动态生成
    file_name: str = None  # 将在 update() 中动态生成
    log_name: str = None  # 将在 update() 中动态生成
    df_name: str = None  # 将在 update() 中动态生成

    def update(self):
        # 验证必需参数
        if self.dataset is None:
            raise ValueError("参数 --dataset 是必需的！例如: --dataset sift")
        if self.k is None:
            raise ValueError("参数 --k 是必需的！例如: --k 10")
        if self.n_bkt is None:
            raise ValueError("参数 --n_bkt 是必需的！例如: --n_bkt 64")
        
        # 验证并标准化 metric 参数
        if self.dis_metric is None or self.dis_metric == '':
            self.dis_metric = 'L2'  # default to L2 distance
        
        # 标准化 metric 名称（支持多种输入格式）
        metric_lower = self.dis_metric.lower()
        if metric_lower in ['l2', 'euclidean', 'euclidean_distance']:
            self.dis_metric = 'L2'
        elif metric_lower in ['ip', 'inner_product', 'dot', 'dot_product']:
            self.dis_metric = 'inner_product'
        else:
            # 保持用户输入的原始值，但给出警告
            print(f"警告: 未知的 metric 值 '{self.dis_metric}'，将使用原始值。支持的值: 'L2', 'inner_product'")
        
        # 动态生成日志路径和文件名（包含 metric 信息）
        self.pth_log = f'/data/tmp/lira/{self.dataset}/ML_kmeans_RE_FLAT/'
        self.file_name = f'{self.dataset}-k={self.k}-ML_kmeans={self.n_bkt}_FLAT_Metric={self.dis_metric}_ReType={self.duplicate_type}_ReRatio={self.redundancy_ratio}'
        self.log_name = f'{self.file_name}.txt'
        self.df_name = f'{self.file_name}.csv'
    
def mul_partition_by_model(data_partition_score, data_predicts, xd_id_sorted_pre, data_2_bkt, cluster_cnts, cluster_ids, begin, end):
    _, n_mul = data_2_bkt.shape
    for t_id in xd_id_sorted_pre[begin:end]:
        cur_c_id = data_2_bkt[t_id, 0] # current cluster id
        sorted_candi_partition = torch.argsort(data_partition_score[t_id], descending=True)
        n_effective_partition = len(torch.where(data_predicts[t_id] != False)[0]) # effective partition id where the probing probability > 0
        n_actual_partition = np.min([n_mul-1, n_effective_partition])
        cur_c_id_loc = torch.where(sorted_candi_partition == cur_c_id)[0] # the location of current cluster in the candidate partition
        if cur_c_id_loc.numel() == 0 or cur_c_id_loc[0] >= n_actual_partition:
            actual_partition = sorted_candi_partition[0 : n_actual_partition].cpu().numpy()
            data_2_bkt[t_id, 1 : n_actual_partition + 1] = actual_partition
        elif n_effective_partition == n_actual_partition:
            actual_partition = sorted_candi_partition[0 : n_actual_partition].cpu().numpy()
            data_2_bkt[t_id, 0 : n_actual_partition] = actual_partition
        else:
            actual_partition = sorted_candi_partition[0 : n_actual_partition + 1].cpu().numpy()
            data_2_bkt[t_id, 0 : n_actual_partition + 1] = actual_partition
        for c_id in actual_partition:
            if c_id != cur_c_id:
                cluster_cnts[c_id] += 1
                cluster_ids[c_id].append(int(t_id))

def cal_metrics(all_predicts, all_targets, epoch, knn_distr_id, cluster_id, results_df, loss, knn=100):
    '''
    Calculation metrics.
    '''
    # metrics for partitions probing
    nprobe_output_mean = torch.mean(torch.sum(all_predicts, dim=1).float()).item()
    nprobe_target_mean = torch.mean(torch.sum(all_targets, dim=1).float()).item()
    accuracy = accuracy_score(all_targets.numpy().flatten(), all_predicts.numpy().flatten())
    hit_rates = torch.logical_and(all_targets, all_predicts).sum(dim=1).float() / all_targets.sum(dim=1).float()  # TP / (TP + FN)
    hit_rate = torch.nanmean(hit_rates).item()  # mean hit rates, using nanmean in case 0

    #  metrics for KNN recall and computations
    n_q = len(all_targets)
    recalls = np.zeros(n_q)
    knn_computations = np.zeros(n_q)
    for qid in range(n_q):
        probing_bids = all_predicts[qid].nonzero(as_tuple=True)[0].numpy()
        if len(probing_bids) == 0:
            # 如果没有预测任何分区，recall 为 0
            recalls[qid] = 0.0
        else:
            aknn = np.unique(np.concatenate([knn_distr_id[qid, q_c] for q_c in probing_bids]))
            recalls[qid] = len(aknn) / knn

    recall_avg = np.mean(recalls)
    cmp_avg = np.mean(knn_computations)

    table = PrettyTable(['Epoch', 'Loss', 'Accuracy', 'Hit Rate', 'nprobe predict', 'nprobe target', 'KNN Recall', 'KNN Computations'])
    table.float_format = "4.4"
    table.add_row([epoch, loss, accuracy, hit_rate, nprobe_output_mean, nprobe_target_mean, recall_avg, cmp_avg])
    fprint(table, fw)
    new_data = pd.DataFrame({
        'Epoch': [epoch],
        'Loss': [loss],
        'Accuracy': [accuracy],
        'Hit Rate': [hit_rate],
        'nprobe predict': [nprobe_output_mean],
        'nprobe target': [nprobe_target_mean],
        'KNN Recall': [recall_avg],
        'KNN Computations': [cmp_avg]
    }).round(4)
    results_df = pd.concat([results_df, new_data], ignore_index=True)

    return results_df

def save_index_artifacts(cfg, kmeans, data_2_bkt, x_d, model, device, fw):
    """
    将 index + 模型相关的内容导出到磁盘，供 C++ search 使用。
    保存内容：
      - centroids.npy           : kmeans.centroids (float32)
      - data_2_bkt.npy         : (N, n_mul) cluster assignment（含冗余，-1 表示无）
      - x_d.npy                : base vectors (可选，看你 C++ 怎么读数据)
      - redundant_flags.npy    : 哪些点被冗余（可选）
      - mlp_2_input.pt         : TorchScript 模型 (LibTorch 可直接加载)

    注：StandardScaler 的 mean / scale 建议在 get_scaled_dist 里另外保存：
      - scaler_mean.npy
      - scaler_scale.npy
    """
    out_dir = cfg.pth_log
    os.makedirs(out_dir, exist_ok=True)

    # 1. centroids
    centroids = kmeans.centroids.astype(np.float32)
    np.save(os.path.join(out_dir, f"{cfg.file_name}_centroids.npy"), centroids)

    # 2. data_2_bkt（含冗余）
    np.save(os.path.join(out_dir, f"{cfg.file_name}_data_2_bkt.npy"),
            data_2_bkt.astype(np.int32))

    # 3. base vectors（可选，方便 C++ 用）
    np.save(os.path.join(out_dir, f"{cfg.file_name}_x_d.npy"),
            x_d.astype(np.float32))

    # 4. 冗余标记（可选）
    #   规则：只要该点在第 1 列以外还有有效 bucket（!= -1），就认为它是冗余点
    redundant_flags = (data_2_bkt[:, 1:] != -1).any(axis=1).astype(np.uint8)
    np.save(os.path.join(out_dir, f"{cfg.file_name}_redundant_flags.npy"),
            redundant_flags)

    # 5. 导出 TorchScript 模型
    model_cpu = model.to('cpu')
    model_cpu.eval()
    scripted_model = torch.jit.script(model_cpu)
    torchscript_path = os.path.join(out_dir, f"{cfg.file_name}_mlp_2_input.pt")
    torch.jit.save(scripted_model, torchscript_path)
    # example_dist = torch.randn(1, cfg.n_bkt, device='cpu')
    # example_vec = torch.randn(1, x_d.shape[1], device='cpu')
    # with torch.no_grad():
    #     scripted = torch.jit.trace(model_cpu, (example_dist, example_vec))
    # torchscript_path = os.path.join(out_dir, f"{cfg.file_name}_mlp_2_input.pt")
    # scripted.save(torchscript_path)

    fprint(f">> [Index] centroids / data_2_bkt / x_d / redundant_flags / mlp_2_input.pt 已保存到 {out_dir}", fw)

# ============================================================================================
# ============================================================================================

if __name__ == "__main__":
    parser = HfArgumentParser(Config)
    cfg = parser.parse_args_into_dataclasses()[0]
    cfg.update()
    n_bkt = cfg.n_bkt

    os.makedirs(cfg.pth_log, exist_ok=True)
    fw = open(cfg.pth_log + cfg.log_name, 'a', encoding='utf-8')
    
    # (1) load data and query
    x_d, x_q, gt_ids = load_data(cfg.dataset, data_path=cfg.data_path)
    if gt_ids is None:
        raise ValueError(f"Ground truth file not found for dataset {cfg.dataset}. Please ensure {cfg.dataset}_groundtruth.ivecs exists.")
    
    fprint(f">> dataset: {cfg.dataset}, data_sizes: {x_d.shape}, query_size: {x_q.shape}, n_bkt: {n_bkt}, knn: {cfg.k}, metric: {cfg.dis_metric}, num_threads: {num_threads}", fw)
    n_d, dim = x_d.shape
    n_q, dim = x_q.shape
    device = f"cuda:{get_idle_gpu()}"
    fprint(f">> device: {device}", fw)
    fprint(f">> distance metric: {cfg.dis_metric}", fw)

    fprint(">> begin data preprocessing (please wait a few minutes)", fw)
    # Compute data self-KNN for training (prefer C++ precomputed cache)
    knn_data = compute_data_knn(x_d, cfg, data_path=cfg.data_path)
    # Use precomputed ground truth for queries
    knn_query = gt_ids[:, :cfg.k]  # Use first k neighbors from ground truth
    fprint(f">> using precomputed ground truth with shape: {knn_query.shape}", fw)

    # (2) initial partitioning
    data_2_bkt = np.full((n_d, cfg.n_mul), -1) # -1 for empty value
    time_start = time.perf_counter()
    kmeans, data_2_single_bkt, cluster_cnts, cluster_ids = build_kmeans_index(x_d, n_bkt)
    data_2_bkt[:, :1] = data_2_single_bkt
    time_build_kmeans = time.perf_counter() - time_start
    fprint(f'>> build kmeans index time: {time_build_kmeans}', fw)

    # (3) probing model
    time_start = time.perf_counter()
    print(">> begin get knn distribution and repartition")
    # knn_distr_cnt_data, knn_distr_id_data = get_knn_distr_redundancy(knn_data, data_2_bkt, cfg)
    labels_data = get_knn_labels_data_only(knn_data, data_2_bkt, cfg)
    time_knn_distr = time.perf_counter() - time_start
    fprint(f'>> get knn distribution time: {time_knn_distr}', fw)
    knn_distr_cnt_query, knn_distr_id_query = get_knn_distr_redundancy(knn_query, data_2_bkt, cfg)
    # labels_data = (knn_distr_cnt_data != 0).astype(np.uint8)
    labels_query = (knn_distr_cnt_query != 0).astype(np.uint8)
    # ------------------------------------------------------------------------------------------
    # observe_knn_tail(knn_distr_cnt, knn_distr_id)
    # ------------------------------------------------------------------------------------------
    # get the distance as input
    time_start = time.perf_counter()
    distances_data_scaled, distances_query_scaled = get_scaled_dist(x_d, x_q, kmeans, n_bkt, cfg)
    time_dist = time.perf_counter() - time_start
    fprint(f'>> get scaled distance time: {time_dist}', fw)
    fprint(f'>> get scaled distance time of queries: {time_dist * n_q / (n_q + n_d)}', fw)

    train_tensor = TensorDataset(torch.tensor(distances_data_scaled, dtype=torch.float32), torch.tensor(x_d, dtype=torch.float32), torch.tensor(labels_data, dtype=torch.float32))
    test_tensor = TensorDataset(torch.tensor(distances_query_scaled, dtype=torch.float32), torch.tensor(x_q, dtype=torch.float32), torch.tensor(labels_query, dtype=torch.float32))
    train_loader = DataLoader(train_tensor, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=cfg.batch_size, shuffle=False)

    # model training and evaluation
    model = MLP_2_Input(input_dim1=n_bkt, input_dim2=dim, output_dim=n_bkt).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch = -1
    time_start = time.perf_counter()
    all_targets, all_predicts, loss_test, all_outputs = model_evaluate(model, test_loader, criterion, device)
    time_test = time.perf_counter() - time_start
    fprint(f'Epoch {epoch}, Test Loss: {loss_test}, time_test: {time_test}', fw)
    pd_cols = ['Epoch', 'Accuracy', 'Hit Rate', 'nprobe predict', 'nprobe target', 'KNN Recall', 'KNN Computations']
    results_df = pd.DataFrame(columns=pd_cols)
    results_df = cal_metrics(all_predicts, all_targets, epoch, knn_distr_id_query, cluster_ids, results_df, loss_test, knn=cfg.k)
    for epoch in range(cfg.n_epoch):
        time_start = time.perf_counter()
        loss_train = model_train(model, train_loader, device, optimizer, criterion)
        time_train = time.perf_counter() - time_start

        time_start = time.perf_counter()
        all_targets, all_predicts, loss_test, all_outputs = model_evaluate(model, test_loader, criterion, device)
        time_test = time.perf_counter() - time_start
        fprint(f'Epoch {epoch}, Train Loss: {loss_train}, Test Loss: {loss_test}, time_train: {time_train}, time_test: {time_test}', fw)
        results_df = cal_metrics(all_predicts, all_targets, epoch, knn_distr_id_query, cluster_ids, results_df, loss_test, knn=cfg.k)

    # (4) redundancy with probing model
    fprint(f">> begin redundancy with {cfg.duplicate_type}, metric: {cfg.dis_metric}", fw)
        # (4) redundancy with probing model
    fprint(f">> begin redundancy with {cfg.duplicate_type}, metric: {cfg.dis_metric}", fw)
    if cfg.duplicate_type == 'model':
        # 使用模型评估所有数据点（保持原有逻辑）
        _, data_predicts, _, data_partition_score = model_evaluate(model, train_loader, criterion, device)
        nprobe_predicts = torch.sum(data_predicts, axis=1)
        xd_id_sorted_pre = torch.argsort(nprobe_predicts, descending=True)
        
        # 计算需要冗余的数据点数量（前 redundancy_ratio % 的数据）
        n_d_model = len(data_predicts)
        n_redundancy = int(n_d_model * cfg.redundancy_ratio)
        fprint(f">> 冗余比例: {cfg.redundancy_ratio * 100:.1f}%, 冗余向量数: {n_redundancy}/{n_d_model}", fw)
        
        # 一次性对前 x% 的数据进行冗余分配
        fprint(f">> 开始对前 {cfg.redundancy_ratio * 100:.1f}% 的向量进行冗余分配...", fw)
        mul_partition_by_model(
            data_partition_score, data_predicts, xd_id_sorted_pre,
            data_2_bkt, cluster_cnts, cluster_ids,
            begin=0, end=n_redundancy
        )

        # 此时 data_2_bkt / cluster_ids 已经包含冗余信息
        # 不在 Python 里做 search / query_tuning，交给 C++ 实现
        fprint(">> 冗余分配完成（Python 端只负责建 index，不做 search）", fw)

        # 保存 index + 模型到磁盘，供 C++ 使用
        save_index_artifacts(cfg, kmeans, data_2_bkt, x_d, model, device, fw)

    elif cfg.duplicate_type == 'None':
        # 不做任何冗余，直接保存 index + 模型
        fprint(">> 不进行冗余（duplicate_type == 'None'），保存原始单分区 index", fw)
        save_index_artifacts(cfg, kmeans, data_2_bkt, x_d, model, device, fw)


    fprint("finish!", fw)
    results_df.to_csv(cfg.pth_log + cfg.df_name, index=False)  # Save DataFrame to CSV
    fw.close()

