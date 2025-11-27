import sys
import numpy as np
import os
import subprocess
import re
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import faiss
faiss.omp_set_num_threads(1)
import torch
import pandas as pd
from tqdm import tqdm
import random
import time
np.random.seed(43)
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

def read_xvecs(file_path, dtype='float32'):
    """
    Read .xvecs format files (fvecs, ivecs, bvecs, etc.)
    
    Args:
        file_path: path to the xvecs file
        dtype: data type ('float32' for fvecs, 'int32' for ivecs, 'uint8' for bvecs)
    
    Returns:
        numpy array of shape (n_vectors, dimension)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    x = np.memmap(file_path, dtype='int32', mode='r')
    d = x[0]
    return x.view(dtype).reshape(-1, d + 1)[:, 1:]

def load_data(dataset_name, data_path='/data/vector_datasets'):
    """
    Load dataset using unified xvecs format
    
    Args:
        dataset_name: name of the dataset (e.g., 'sift', 'spacev10m')
        data_path: base path where datasets are stored
    
    Returns:
        x_d: base vectors (numpy array)
        x_q: query vectors (numpy array)
        gt_ids: ground truth nearest neighbors (numpy array or None)
    """
    dataset_dir = os.path.join(data_path, dataset_name)
    
    dtype = 'float32'
    ext = 'fvecs'
    
    # Load base vectors
    base_file = os.path.join(dataset_dir, f'{dataset_name}_base.{ext}')
    if not os.path.exists(base_file):
        # Try alternative naming: some datasets might use different suffixes
        base_file = os.path.join(dataset_dir, f'{dataset_name}_learn.{ext}')
    
    x_d = read_xvecs(base_file, dtype=dtype)
    
    # Load query vectors
    query_file = os.path.join(dataset_dir, f'{dataset_name}_query.{ext}')
    x_q = read_xvecs(query_file, dtype=dtype)
    
    # Load ground truth if exists
    gt_file = os.path.join(dataset_dir, f'{dataset_name}_groundtruth.ivecs')
    if os.path.exists(gt_file):
        gt_ids = read_xvecs(gt_file, dtype='int32')
    else:
        gt_ids = None
    
    # Ensure contiguous arrays for faiss
    x_d = np.ascontiguousarray(x_d)
    x_q = np.ascontiguousarray(x_q)
    
    print(f"Loaded dataset '{dataset_name}':")
    print(f"  Base vectors: {x_d.shape}")
    print(f"  Query vectors: {x_q.shape}")
    if gt_ids is not None:
        print(f"  Ground truth: {gt_ids.shape}")
    
    return x_d, x_q, gt_ids

def get_idle_gpu():
    # get GPU status
    smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']).decode()
    gpu_usage = [int(re.search(r'\d+$', line).group()) for line in smi_output.strip().split('\n')]
    idle_gpu_index = gpu_usage.index(min(gpu_usage))

    return idle_gpu_index

def get_dist_cid(data, kmeans, n_bkt):
    """
    批量计算数据点到聚类中心的距离
    使用 scipy.spatial.distance.cdist 优化内存和速度
    """
    # 使用更大的批量大小以提高效率
    batch_size = 10000
    n_samples = data.shape[0]
    n_centroids = kmeans.centroids.shape[0]
    
    # 预分配结果数组
    distances = np.zeros((n_samples, n_centroids), dtype=np.float32)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        data_batch = data[start_idx:end_idx]
        # 使用 cdist 计算欧氏距离，比 broadcasting 更节省内存
        distances_batch = cdist(data_batch, kmeans.centroids, metric='euclidean').astype(np.float32)
        distances[start_idx:end_idx] = distances_batch

    return distances

def get_scaled_dist(x_d, x_q, kmeans, n_bkt):
    n_d = x_d.shape[0]
    n_q = x_q.shape[0]

    if n_d < 1_000_000:
        distances_data = get_dist_cid(x_d, kmeans, n_bkt).astype(np.float32)
        distances_query = get_dist_cid(x_q, kmeans, n_bkt).astype(np.float32)
        scaler = StandardScaler()
        distances_data_scaled = scaler.fit_transform(distances_data).astype(np.float16)   # <- 改为 f16
        distances_query_scaled = scaler.transform(distances_query).astype(np.float16)     # <- 改为 f16
        return distances_data_scaled, distances_query_scaled

    # 大数据集：两遍法 + memmap(f16)
    print(f">> 计算距离（数据集大小: {n_d}, 使用增量方式）...")
    batch_size = 10000
    scaler = StandardScaler()

    # pass1: partial_fit
    for start_idx in range(0, n_d, batch_size):
        end_idx = min(start_idx + batch_size, n_d)
        distances_batch = get_dist_cid(x_d[start_idx:end_idx], kmeans, n_bkt).astype(np.float32)
        scaler.partial_fit(distances_batch)
        del distances_batch

    # pass2: transform + 写入 memmap(float16)
    os.makedirs('/tmp/lira_cache', exist_ok=True)
    mm_path = '/tmp/lira_cache/distances_data_scaled.f16.memmap'
    distances_data_scaled = np.memmap(mm_path, dtype='float16', mode='w+', shape=(n_d, n_bkt))  # <- memmap

    for start_idx in range(0, n_d, batch_size):
        end_idx = min(start_idx + batch_size, n_d)
        distances_batch = get_dist_cid(x_d[start_idx:end_idx], kmeans, n_bkt).astype(np.float32)
        distances_data_scaled[start_idx:end_idx] = scaler.transform(distances_batch).astype(np.float16)  # <- f16
        del distances_batch

    # 查询集一次性处理，也转 f16（通常较小）
    distances_query = get_dist_cid(x_q, kmeans, n_bkt).astype(np.float32)
    distances_query_scaled = scaler.transform(distances_query).astype(np.float16)

    # 注意：返回的是 memmap（行为与 ndarray 一致），后续可直接 torch.tensor(...)
    return distances_data_scaled, distances_query_scaled

def get_scaled_dist_data(x_d, kmeans, n_bkt):
    """
    只计算并标准化数据到聚类中心的距离
    对于大数据集，使用批量处理以节省内存
    """
    n_d = x_d.shape[0]
    
    # 如果数据集较小（<50万），直接计算
    if n_d < 500000:
        distances_data = get_dist_cid(x_d, kmeans, n_bkt)
        scaler = StandardScaler()
        distances_data_scaled = scaler.fit_transform(distances_data)
        return distances_data_scaled
    
    # 对于大数据集，使用增量方式
    batch_size = 50000
    scaler = StandardScaler()
    
    # 第一遍：fit scaler
    for start_idx in range(0, n_d, batch_size):
        end_idx = min(start_idx + batch_size, n_d)
        distances_batch = get_dist_cid(x_d[start_idx:end_idx], kmeans, n_bkt)
        scaler.partial_fit(distances_batch)
        del distances_batch
    
    # 第二遍：transform
    distances_data_scaled = np.zeros((n_d, n_bkt), dtype=np.float32)
    for start_idx in range(0, n_d, batch_size):
        end_idx = min(start_idx + batch_size, n_d)
        distances_batch = get_dist_cid(x_d[start_idx:end_idx], kmeans, n_bkt)
        distances_data_scaled[start_idx:end_idx] = scaler.transform(distances_batch).astype(np.float32)
        del distances_batch
    
    return distances_data_scaled

def fprint(message, file=None):
    print(message)  # print to terminal
    if file:
        print(message, file=file)  # print to file


def compute_data_knn(x_data, cfg, data_path='/data/vector_datasets'):
    '''
    Load or compute KNN of data on itself (for training data)
    
    Priority:
      1. Load from C++ precomputed cache (.bin file) - RECOMMENDED
      2. Load from Python cache (.npy file)
      3. Compute using simple FLAT index (slow, not recommended for large datasets)
    
    Args:
        x_data: data vectors
        cfg: configuration object
        data_path: base path for datasets
    '''
    # Create cache directory if not exists
    cache_dir = os.path.join(data_path, cfg.dataset, 'knn_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    n_data = len(x_data)
    
    # Try to load from C++ precomputed cache first (.bin format)
    # Try IVF approximate cache first (faster to compute), then exact cache
    cache_patterns = [
        f'{cfg.dataset}-data_self_knn{cfg.k}-n{n_data}_ivf_nprobe*.bin',  # IVF with any nprobe
        f'{cfg.dataset}-data_self_knn{cfg.k}-n{n_data}.bin',              # Exact search
    ]
    
    cache_file_bin = None
    for pattern in cache_patterns:
        import glob
        matches = glob.glob(os.path.join(cache_dir, pattern))
        if matches:
            # Use the most recently created file
            cache_file_bin = max(matches, key=os.path.getctime)
            break
    
    cache_file_npy = os.path.join(cache_dir, f'{cfg.dataset}-data_self_knn{cfg.k}-n{n_data}.npy')
    
    # Check for C++ generated .bin file first (RECOMMENDED)
    if cache_file_bin and os.path.exists(cache_file_bin):
        print(f"✓ Loading precomputed KNN from C++ cache: {os.path.basename(cache_file_bin)}")
        knn_data = np.fromfile(cache_file_bin, dtype=np.int32).reshape(n_data, cfg.k)
        print(f"✓ Loaded KNN with shape: {knn_data.shape}")
        return knn_data
    
    # Check for Python generated .npy file
    if os.path.exists(cache_file_npy):
        print(f"✓ Loading cached KNN from: {cache_file_npy}")
        knn_data = np.load(cache_file_npy).astype(int)
        return knn_data
    
    # No cache found - compute using Python (slow, not recommended)
    print(f"\n{'='*60}")
    print(f"WARNING: No precomputed KNN cache found!")
    print(f"{'='*60}")
    print(f"Computing KNN in Python is SLOW for large datasets.")
    print(f"Recommendation: Use C++ program for faster computation:")
    print(f"  # Fast approximate search (recommended):")
    print(f"  ./compute_knn {cfg.dataset} {data_path} {cfg.k} 64 24")
    print(f"  # Or exact search (slower):")
    print(f"  ./compute_knn {cfg.dataset} {data_path} {cfg.k} 0 24")
    print(f"{'='*60}\n")
    
    dim = x_data.shape[1]
    print(f"Computing data self-KNN for {n_data} vectors (dim={dim})...")
    print(f"This may take a while... Please wait or interrupt and use C++ program.")
    
    t_start = time.time()
    
    # Simple FLAT index (exact search)
    if cfg.dis_metric == 'inner_product':
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    
    index.add(x_data)
    
    # Search in batches to avoid memory issues
    batch_size = min(10000, n_data)
    knn_data = np.zeros((n_data, cfg.k), dtype=np.int32)
    
    print(f"Searching in batches of {batch_size}...")
    for start_idx in tqdm(range(0, n_data, batch_size), desc="Computing KNN"):
        end_idx = min(start_idx + batch_size, n_data)
        batch = x_data[start_idx:end_idx]
        _, batch_knn = index.search(batch, cfg.k + 1)  # +1 to exclude self
        # Remove self (first neighbor is usually itself)
        knn_data[start_idx:end_idx] = batch_knn[:, 1:cfg.k+1]
    
    t_elapsed = time.time() - t_start
    print(f"KNN computation completed in {t_elapsed:.2f}s")
    
    # Save to cache
    np.save(cache_file_npy, knn_data)
    print(f"✓ Cached to: {cache_file_npy}")
    
    return knn_data

def build_kmeans_index(x_data, n_bkt):
    n_d, dim = x_data.shape
    kmeans = faiss.Kmeans(dim, n_bkt, niter=20, verbose=True)
    kmeans.train(x_data)
    _, data_2_bkt = kmeans.index.search(x_data, 1)  # get the cluster index of each data point
    cluster_cnts = np.bincount(data_2_bkt.flatten())  # get the number of data points in each cluster
    cluster_ids = [[] for _ in range(n_bkt)]
    for i in range(n_d):
        cluster_ids[data_2_bkt[i, 0]].append(i)
    return kmeans, data_2_bkt, cluster_cnts, cluster_ids

def get_knn_distr(knn, data_2_bkt, cfg):
    '''
    knn_distr. the distribution of knn counts of each query
    knn_id. the knn ids of each query in each cluster
    '''
    n_data = knn.shape[0]
    knn_distr_cnt = np.zeros((n_data, cfg.n_bkt), dtype=int)
    knn_distr_id = np.empty((n_data, cfg.n_bkt), dtype=object)
    for i in range(n_data):
        for j in range(cfg.n_bkt):
            knn_distr_id[i, j] = []
    
    for v_idx in tqdm(range(n_data)):
        v_knn_ids = knn[v_idx] # the knn ids of a query
        v_knn_bkts = data_2_bkt[v_knn_ids].flatten() # the cluster id of each knn of a query
        unique_bkts, counts = np.unique(v_knn_bkts, return_counts=True)
        knn_distr_cnt[v_idx, unique_bkts] = counts
        for bkt in unique_bkts:
            knn_distr_id[v_idx, bkt] = v_knn_ids[v_knn_bkts == bkt].tolist()
    
    return knn_distr_cnt, knn_distr_id

def get_knn_distr_redundancy(knn, data_2_bkt, cfg):
    '''
    knn_distr. the distribution of knn counts of each query
    knn_id. the knn ids of each query in each cluster
    '''
    _, n_mul = data_2_bkt.shape
    n_data = knn.shape[0]
    knn_distr_cnt = np.zeros((n_data, cfg.n_bkt), dtype=int)
    knn_distr_id = np.empty((n_data, cfg.n_bkt), dtype=object)
    for i in range(n_data):
        for j in range(cfg.n_bkt):
            knn_distr_id[i, j] = []
    
    for v_idx in tqdm(range(n_data)):
        v_knn_ids = knn[v_idx] # the knn ids of a query
        v_knn_bkts = data_2_bkt[v_knn_ids].flatten() # the cluster id of each knn of a query
        unique_bkts, counts = np.unique(v_knn_bkts, return_counts=True)
        if unique_bkts[0] == -1:
            unique_bkts = unique_bkts[1:]
            counts = counts[1:]
        knn_distr_cnt[v_idx, unique_bkts] = counts
        for bkt in unique_bkts:
            v_mul_knn_ids = np.repeat(v_knn_ids, n_mul) # [1,3,...] -> [1,...,1,3,...,3,...]
            knn_distr_id[v_idx, bkt] = v_mul_knn_ids[v_knn_bkts == bkt].tolist()
    
    return knn_distr_cnt, knn_distr_id

def get_knn_labels_data_only(knn, data_2_bkt, cfg):
    """
    只为 data 端生成 0/1 label 矩阵：
      labels_data[i, bkt] = 1 表示样本 i 在 bucket bkt 里至少有一个 knn
    不返回 knn_distr_id，不返回完整计数矩阵，避免巨大的 object 数组。
    """
    n_data = knn.shape[0]
    _, n_mul = data_2_bkt.shape
    n_bkt = cfg.n_bkt

    # 直接生成 0/1，dtype=uint8
    labels = np.zeros((n_data, n_bkt), dtype=np.uint8)

    for v_idx in tqdm(range(n_data), desc="building labels_data"):
        v_knn_ids = knn[v_idx]                     # shape (k,)
        # 每个 knn 映射的 bucket（考虑冗余）
        v_knn_bkts = data_2_bkt[v_knn_ids].flatten()  # shape (k * n_mul,)
        unique_bkts = np.unique(v_knn_bkts)
        # 去掉 -1（没填充的冗余槽）
        unique_bkts = unique_bkts[unique_bkts != -1]
        if unique_bkts.size == 0:
            continue
        labels[v_idx, unique_bkts] = 1

    return labels

def create_flat_indexes(x_d, xd_id_bkts, cfg, dis_metric: str = 'L2'):
    """
    create a flat index for each bucket
    """
    flat_indexes = []
    for i in range(cfg.n_bkt):
        xd_id_bkt = np.array(xd_id_bkts[i])
        xd_bkt = x_d[xd_id_bkt]
        if dis_metric == 'inner_product':
            flat_index = faiss.IndexFlatIP(xd_bkt.shape[1])
        else:
            flat_index = faiss.IndexFlatL2(xd_bkt.shape[1])
        flat_index.add(xd_bkt)
        flat_indexes.append(flat_index)

    return flat_indexes

def create_inner_indexes(x_d, cluster_ids, cfg):
    """
    create FLAT indexes for each bucket
    """
    inner_indexes = create_flat_indexes(x_d, cluster_ids, cfg, dis_metric=cfg.dis_metric)
    return inner_indexes

def min_exclude_zero(row):
    non_zero_elements = row[row != 0]
    if non_zero_elements.size > 0:
        return non_zero_elements.min()
    else:
        return np.nan

def observe_knn_tail(knn_distr_cnt, knn_distr_id, n_d, cfg, model, distances_data_scaled, x_d, data_2_bkt, device):
    '''
    observe the distribution of long-tail knn
    '''
    min_values = np.apply_along_axis(min_exclude_zero, 1, knn_distr_cnt)
    unique_elements, counts = np.unique(min_values, return_counts=True)
    print(np.asarray((unique_elements, counts)).T)

    # analyze the long-tail data. When a data is a long-tail data in a knn distribution, get the replica buckets where # of knn >= 2
    tail_ids = []
    tail_id_other_bkts = np.zeros((n_d, cfg.n_bkt), dtype=bool)
    for q_id in range(len(knn_distr_cnt)):
        bkt_tail = np.where(knn_distr_cnt[q_id] == 1)[0] # the bucket with only one knn
        if len(bkt_tail) > 0:
            bkt_non_tail = np.where(knn_distr_cnt[q_id] > 1)[0] # the bucket with more than one knn
            vec_tail = np.concatenate(knn_distr_id[q_id][bkt_tail]) # the long-tail data id
            tail_ids.extend(vec_tail)
            for vec in vec_tail:
                tail_id_other_bkts[vec][bkt_non_tail] = 1
    
    tail_id = np.where(np.any(tail_id_other_bkts, axis=1))[0] # the long-tail data id after removing duplicates
    # np.save(f'./dataset/{cfg.dataset}/{cfg.dataset}-bkt{cfg.n_bkt}query_longtail_id.npy', tail_id_test)
    # np.save(f'./dataset/{cfg.dataset}/{cfg.dataset}-bkt{cfg.n_bkt}data_longtail_id.npy', tail_id_train)
    
    n_tail_id = len(tail_id) # 5, len(tail_id)
    output_rank_replica = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the ranking of replica buckets by model output (probing rank)
    dist_rank_replica = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the ranking of replica buckets by centroids distance in IVF/kmeans (distance rank)
    output_matrix = np.zeros((n_tail_id, cfg.n_bkt)) # model output
    replica_matrix = np.zeros((n_tail_id, cfg.n_bkt), dtype=int) # the replica buckets for each long-tail data

    for idx, vid in enumerate(tail_id[:n_tail_id]):
        model.eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(distances_data_scaled[vid]).unsqueeze(0).to(device), 
                torch.from_numpy(x_d[vid]).unsqueeze(0).to(device)
            ).cpu()[0]
        output_matrix[idx] = output
        vec_tail_bkt = np.where(tail_id_other_bkts[vid] == True)[0]
        replica_matrix[idx][vec_tail_bkt] = 1
        output_sorted_idx = np.argsort(-output) # model output sorted index (probing rank)
        distance_sorted_idx = np.argsort(distances_data_scaled[vid]) # centroids distance sorted in ivf/kmeans (distance rank)
        bkt_output_pairs = [(bkt, output[bkt].item(), np.where(output_sorted_idx == bkt)[0][0], np.where(distance_sorted_idx == bkt)[0][0]) for bkt in vec_tail_bkt]
        sorted_bkt_output_pairs = sorted(bkt_output_pairs, key=lambda x: x[2])
        
        print('-' * 40)
        print(f'vec [{vid}] prob of tail_to_replica_bkt, self bkt_id {data_2_bkt[vid]}')
        for bkt, probi, p_rank, dis_rank in sorted_bkt_output_pairs:
            output_rank_replica[idx][p_rank] = 1
            dist_rank_replica[idx][dis_rank] = 1
            print(f"bkt_Id: {bkt}, output: {probi:.4f}, output rank: {p_rank}, dist rank: {dis_rank}")

    # analysis of the nprobe reduction when putting the long-tail data into the replica buckets with probing rank and distance rank
    output_rank_replica_cum = np.maximum.accumulate(output_rank_replica, axis=1)
    dist_rank_replica_cum = np.maximum.accumulate(dist_rank_replica, axis=1)

    output_rank_valid = np.sum(output_rank_replica_cum, axis=0) / n_tail_id # the effectiveness of long-tail data
    dist_rank_valid = np.sum(dist_rank_replica_cum, axis=0) / n_tail_id
    print("output_rank_valid", output_rank_valid)
    print("dist_rank_valid", dist_rank_valid)

    output_matrix_masked = output_matrix * replica_matrix
    replica_values = output_matrix_masked[output_matrix_masked != 0]

def per_query(all_outputs, knn_distr_cnt_query, cluster_cnts, n_bkt, cfg):
    nq_test = 100
    cmp_per_q = np.zeros(nq_test, dtype=int)
    nprobe_per_q = np.zeros(nq_test, dtype=int)
    recall_target = 0.98

    df_result_perquery = pd.DataFrame(columns=['q_id', 'nprobe', 'cmp'])

    for q_id in range(nq_test):
        for probeM in range(1, 20):
            Mbkt = all_outputs[q_id].topk(probeM).indices.cpu().numpy()
            match_count = knn_distr_cnt_query[q_id, Mbkt].sum()
            if match_count / cfg.k >= recall_target:
                cmp_per_q[q_id] = cluster_cnts[Mbkt].sum()
                nprobe_per_q[q_id] = probeM
                break
        df_result_perquery = pd.concat([df_result_perquery, pd.DataFrame({'q_id': [q_id], 'nprobe': [nprobe_per_q[q_id]], 'cmp': [cmp_per_q[q_id]]})], ignore_index=True)
    df_result_perquery.to_csv(cfg.pth_log + f'{cfg.dataset}-k={cfg.k}-ML_kmeans={n_bkt}_perquery.csv', index=False)


