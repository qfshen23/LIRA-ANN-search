'''
LIRA for small-scale datasets
1. Redundancy can be set through repa_step
2. Internal index uses FLAT for exact search within each partition.
cluster, bucket, partition are used interchangeably.
'''
import numpy as np
import os
num_threads = 1
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
        
        if self.dis_metric == None:
            self.dis_metric = 'L2'  # default to L2 distance
        
        # 动态生成日志路径和文件名
        self.pth_log = f'./logs/{self.dataset}/ML_kmeans_RE_FLAT/'
        self.file_name = f'{self.dataset}-k={self.k}-ML_kmeans={self.n_bkt}_FLAT_ReType={self.duplicate_type}_ReRatio={self.redundancy_ratio}'
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


def get_cmp_recall(inner_indexes, x_q, xd_id_bkt, cfg):
    """
    search in each partition for each query
    return the search time, distance comparisons, and found aknn id (which need to be deduplicated later)
    """
    n_bkt = cfg.n_bkt
    n_q = len(x_q)
    search_time = np.zeros((n_q, n_bkt))  # record the search time
    cmp_distr_all = np.zeros((n_q, n_bkt), dtype=int)  # record the distance comparisons
    found_aknn_id = np.full((n_q, n_bkt, cfg.k), -1)  # record the found aknn id
    xd_id_bkt_int = [np.array(bkt).astype(int) for bkt in xd_id_bkt]
    
    # loop for each bucket
    for b_id in tqdm(range(n_bkt)): 
        inner_index = inner_indexes[b_id]
        xd_id_bid = xd_id_bkt_int[b_id] # the data id in the current bucket
        if inner_index is None or len(xd_id_bkt_int[b_id]) == 0:
            continue
        
        # loop for each query
        for q_id in range(n_q):
            q = x_q[q_id].reshape(1, -1)
            t_start = time.time()
            _, idx_in_bkt = inner_index.search(q, cfg.k)
            aknn_id = xd_id_bid[idx_in_bkt.reshape(-1)]
            found_aknn_id[q_id][b_id] = aknn_id
            cmp_distr_all[q_id][b_id] = inner_index.ntotal
            search_time[q_id][b_id] = time.time() - t_start

    return search_time, cmp_distr_all, found_aknn_id

def query_tuning(all_outputs, knn_distr_id, found_aknn_id, search_time, cfg, fw, part=0):
    n_q = len(all_outputs)
    n_s = n_q
    
    pd_cols = ['threshold', 'nprobe', 'Recall', 'Computations', 'QPS']
    df_threshold = pd.DataFrame(columns=pd_cols)
    
    # 创建目录
    os.makedirs(cfg.pth_log + cfg.file_name + f'_tuning_threshold/', exist_ok=True)
    
    # 写入标题到主日志
    fprint("", fw)
    fprint("=" * 90, fw)
    fprint(f"Query Tuning Results - Part {part}", fw)
    fprint("=" * 90, fw)
    fprint(f"Dataset: {cfg.dataset}, n_bkt: {cfg.n_bkt}, redundancy_ratio: {cfg.redundancy_ratio}", fw)
    fprint(f"Number of queries: {n_q}", fw)
    fprint("=" * 90, fw)
    
    # 创建表格
    table = PrettyTable(['Threshold', 'nprobe', 'Recall', 'Computations', 'QPS'])
    table.float_format = "4.4"
    
    for threshold in np.arange(0.1, 1.0, 0.02):
        thre_recall = np.zeros(n_s)
        thre_cmp = np.zeros(n_s)
        thre_nprobe = np.zeros(n_s)
        thre_time = np.zeros(n_s)  # 每个查询的搜索时间
        for i in range(n_s):
            # get the bucket with probing probability > threshold
            top_m_indices = np.where(all_outputs[i] > threshold)[0]
            thre_nprobe[i] = len(top_m_indices)
            thre_cmp[i] = np.sum(cmp_distr_all[i, top_m_indices])
            thre_time[i] = np.sum(search_time[i, top_m_indices])  # 累加该查询在各分区的搜索时间
            found_knn = set()
            for q_c in top_m_indices:
                aknn_c = set(knn_distr_id[i][q_c]).intersection(found_aknn_id[i][q_c])
                found_knn.update(aknn_c)
            thre_recall[i] = len(found_knn) / cfg.k

        thre_recall_avg = np.mean(thre_recall)
        thre_cmp_avg = np.mean(thre_cmp)
        thre_nprobe_avg = np.mean(thre_nprobe)
        thre_time_avg = np.mean(thre_time)  # 平均查询时间
        thre_qps = 1.0 / thre_time_avg if thre_time_avg > 0 else 0.0  # QPS = 1 / 平均查询时间
        
        # 打印到控制台
        print(f'threshold: {threshold:.3f}, nprobe: {thre_nprobe_avg:.2f}, Recall: {thre_recall_avg:.4f}, Computations: {thre_cmp_avg:.0f}, QPS: {thre_qps:.2f}')
        
        # 添加到表格
        table.add_row([threshold, thre_nprobe_avg, thre_recall_avg, thre_cmp_avg, thre_qps])
        
        # 保存到 DataFrame
        new_data = pd.DataFrame([{'threshold': threshold, 'nprobe': thre_nprobe_avg, 'Recall': thre_recall_avg, 'Computations': thre_cmp_avg, 'QPS': thre_qps}])
        df_threshold = pd.concat([df_threshold, new_data], ignore_index=True)
    
    # 写入表格到主日志文件
    fprint(table, fw)
    fprint("=" * 90, fw)
    fprint("", fw)
    
    # 保存 CSV
    csv_path = cfg.pth_log + cfg.file_name + f'_tuning_threshold/{cfg.duplicate_type}_{part}.csv'
    df_threshold.to_csv(csv_path, index=False)
    
    fprint(f">> Query tuning CSV saved to: {csv_path}", fw)

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
    
    fprint(f">> dataset: {cfg.dataset}, data_sizes: {x_d.shape}, query_size: {x_q.shape}, n_bkt: {n_bkt}, knn: {cfg.k}, num_threads: {num_threads}", fw)
    n_d, dim = x_d.shape
    n_q, dim = x_q.shape
    device = f"cuda:{get_idle_gpu()}"
    fprint(f">> device: {device}", fw)

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
    knn_distr_cnt_data, knn_distr_id_data = get_knn_distr_redundancy(knn_data, data_2_bkt, cfg)
    time_knn_distr = time.perf_counter() - time_start
    fprint(f'>> get knn distribution time: {time_knn_distr}', fw)
    knn_distr_cnt_query, knn_distr_id_query = get_knn_distr_redundancy(knn_query, data_2_bkt, cfg)
    labels_data = (knn_distr_cnt_data != 0).astype(np.uint8)
    labels_query = (knn_distr_cnt_query != 0).astype(np.uint8)
    # ------------------------------------------------------------------------------------------
    # observe_knn_tail(knn_distr_cnt, knn_distr_id)
    # ------------------------------------------------------------------------------------------
    # get the distance as input
    time_start = time.perf_counter()
    distances_data_scaled, distances_query_scaled = get_scaled_dist(x_d, x_q, kmeans, n_bkt)
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
    fprint(f">> begin redundancy with {cfg.duplicate_type}", fw)
    if cfg.duplicate_type == 'model':
        # 使用模型评估所有数据点
        _, data_predicts, _, data_partition_score = model_evaluate(model, train_loader, criterion, device)
        nprobe_predicts = torch.sum(data_predicts, axis=1)
        xd_id_sorted_pre = torch.argsort(nprobe_predicts, descending=True)
        
        # 计算需要冗余的数据点数量（前 redundancy_ratio % 的数据）
        n_d = len(data_predicts)
        n_redundancy = int(n_d * cfg.redundancy_ratio)
        fprint(f">> 冗余比例: {cfg.redundancy_ratio * 100:.1f}%, 冗余向量数: {n_redundancy}/{n_d}", fw)
        
        # 先测试无冗余的基准性能
        fprint(">> 测试基准性能（无冗余）...", fw)
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, search_time, cfg, fw, part=0)
        
        # 一次性对前 x% 的数据进行冗余分配
        fprint(f">> 开始对前 {cfg.redundancy_ratio * 100:.1f}% 的向量进行冗余分配...", fw)
        mul_partition_by_model(data_partition_score, data_predicts, xd_id_sorted_pre, 
                              data_2_bkt, cluster_cnts, cluster_ids, 
                              begin=0, end=n_redundancy)
        
        # 更新 KNN 分布
        knn_distr_cnt_query, knn_distr_id_query = get_knn_distr_redundancy(knn_query, data_2_bkt, cfg)
        
        # 重建索引并测试性能
        fprint(f">> 测试冗余后性能（冗余了 {n_redundancy} 个向量）...", fw)
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, search_time, cfg, fw, part=1)
    elif cfg.duplicate_type == 'None':
        # build inner indexes
        time_start = time.perf_counter()
        inner_indexes = create_inner_indexes(x_d, cluster_ids, cfg)
        time_build_flat = time.perf_counter() - time_start
        fprint(f'>> build flat index time: {time_build_flat}', fw)

        time_start = time.perf_counter()
        search_time, cmp_distr_all, found_aknn_id = get_cmp_recall(inner_indexes, x_q, cluster_ids, cfg)
        time_search = time.perf_counter() - time_start
        fprint(f'>> search time: {time_search}', fw)
        query_tuning(all_outputs, knn_distr_id_query, found_aknn_id, search_time, cfg, fw)

    fprint("finish!", fw)
    results_df.to_csv(cfg.pth_log + cfg.df_name, index=False)  # Save DataFrame to CSV
    fw.close()

