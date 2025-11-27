#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <omp.h>

#include <cnpy.h>          // https://github.com/rogersce/cnpy
#include <torch/script.h>  // LibTorch

// =========================== 参数解析 ===========================

struct Args {
    std::string dataset;
    std::string data_path = "/data/vector_datasets";
    std::string artifacts_dir = ".";
    std::string prefix;        // 对应 Python 的 cfg.file_name
    std::string metric = "L2"; // "L2" or "inner_product"
    int k = 10;                // Recall@k
    int num_threads = 32;

    // threshold sweep: [t_min, t_max] with step t_step
    float t_min  = 0.02f;
    float t_max  = 0.80f;
    float t_step = 0.02f;
};

void print_usage() {
    std::cout << "Usage:\n"
              << "  mlp_search \\\n"
              << "    --dataset <name> \\\n"
              << "    --data_path <path_to_datasets_root> \\\n"
              << "    --artifacts_dir <path_to_python_artifacts> \\\n"
              << "    --prefix <file_prefix_same_as_cfg.file_name> \\\n"
              << "    --k <K_eval> \\\n"
              << "    --metric <L2|inner_product> \\\n"
              << "    [--num_threads N] \\\n"
              << "    [--t_min v --t_max v --t_step v]\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string& flag) {
            return a == flag && (i + 1 < argc);
        };
        if (need("--dataset")) {
            args.dataset = argv[++i];
        } else if (need("--data_path")) {
            args.data_path = argv[++i];
        } else if (need("--artifacts_dir")) {
            args.artifacts_dir = argv[++i];
        } else if (need("--prefix")) {
            args.prefix = argv[++i];
        } else if (need("--k")) {
            args.k = std::stoi(argv[++i]);
        } else if (need("--metric")) {
            args.metric = argv[++i];
        } else if (need("--num_threads")) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (need("--t_min")) {
            args.t_min = std::stof(argv[++i]);
        } else if (need("--t_max")) {
            args.t_max = std::stof(argv[++i]);
        } else if (need("--t_step")) {
            args.t_step = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete arg: " << a << "\n";
            return false;
        }
    }
    if (args.dataset.empty() || args.prefix.empty()) {
        std::cerr << "Error: --dataset and --prefix are required.\n";
        return false;
    }
    return true;
}

// =========================== fvecs / ivecs 读取 ===========================

std::vector<float> read_fvecs(const std::string& fname, size_t& n, size_t& d) {
    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open fvecs file: " + fname);
    }
    int dim;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Error reading dim from: " + fname);
    }
    d = static_cast<size_t>(dim);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long record_size = sizeof(int) + d * sizeof(float);
    if (file_size % record_size != 0) {
        fclose(f);
        throw std::runtime_error("Invalid fvecs file size: " + fname);
    }
    n = file_size / record_size;
    fseek(f, 0, SEEK_SET);

    std::vector<float> data(n * d);
    for (size_t i = 0; i < n; ++i) {
        int dim_i;
        if (fread(&dim_i, sizeof(int), 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("Error reading dim for vector " + std::to_string(i));
        }
        if (dim_i != (int)d) {
            fclose(f);
            throw std::runtime_error("Inconsistent dim in fvecs: " + fname);
        }
        if (fread(data.data() + i * d, sizeof(float), d, f) != (size_t)d) {
            fclose(f);
            throw std::runtime_error("Error reading data for vector " + std::to_string(i));
        }
    }
    fclose(f);
    return data;
}

std::vector<int> read_ivecs(const std::string& fname, size_t& n, size_t& d) {
    FILE* f = fopen(fname.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open ivecs file: " + fname);
    }
    int dim;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        fclose(f);
        throw std::runtime_error("Error reading dim from: " + fname);
    }
    d = static_cast<size_t>(dim);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long record_size = sizeof(int) + d * sizeof(int);
    if (file_size % record_size != 0) {
        fclose(f);
        throw std::runtime_error("Invalid ivecs file size: " + fname);
    }
    n = file_size / record_size;
    fseek(f, 0, SEEK_SET);

    std::vector<int> data(n * d);
    for (size_t i = 0; i < n; ++i) {
        int dim_i;
        if (fread(&dim_i, sizeof(int), 1, f) != 1) {
            fclose(f);
            throw std::runtime_error("Error reading dim for vector " + std::to_string(i));
        }
        if (dim_i != (int)d) {
            fclose(f);
            throw std::runtime_error("Inconsistent dim in ivecs: " + fname);
        }
        if (fread(data.data() + i * d, sizeof(int), d, f) != (size_t)d) {
            fclose(f);
            throw std::runtime_error("Error reading data for vector " + std::to_string(i));
        }
    }
    fclose(f);
    return data;
}

// =========================== npy 读取 ===========================

std::vector<float> load_npy_float2d(const std::string& path, size_t& n, size_t& d) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        throw std::runtime_error("Expected float32 npy: " + path);
    }
    if (arr.shape.size() != 2) {
        throw std::runtime_error("Expected 2D npy: " + path);
    }
    n = arr.shape[0];
    d = arr.shape[1];
    float* data_ptr = arr.data<float>();
    std::vector<float> out(n * d);
    std::memcpy(out.data(), data_ptr, n * d * sizeof(float));
    return out;
}

std::vector<int> load_npy_int2d(const std::string& path, size_t& n, size_t& d) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(int32_t)) {
        throw std::runtime_error("Expected int32 npy: " + path);
    }
    if (arr.shape.size() != 2) {
        throw std::runtime_error("Expected 2D npy: " + path);
    }
    n = arr.shape[0];
    d = arr.shape[1];
    int32_t* data_ptr = arr.data<int32_t>();
    std::vector<int> out(n * d);
    std::memcpy(out.data(), data_ptr, n * d * sizeof(int32_t));
    return out;
}

std::vector<float> load_npy_float1d(const std::string& path, size_t& n) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float)) {
        throw std::runtime_error("Expected float32 npy: " + path);
    }
    if (arr.shape.size() != 1) {
        throw std::runtime_error("Expected 1D npy: " + path);
    }
    n = arr.shape[0];
    float* data_ptr = arr.data<float>();
    std::vector<float> out(n);
    std::memcpy(out.data(), data_ptr, n * sizeof(float));
    return out;
}

// =========================== 距离 & 标准化 ===========================

// query -> all centroids 的欧氏距离（和 Python 一致，用 sqrt）
void compute_l2_to_centroids(const float* query,
                             const float* centroids,
                             size_t n_bkt,
                             size_t dim,
                             std::vector<float>& out_dist) {
    out_dist.assign(n_bkt, 0.0f);
    for (long c = 0; c < (long)n_bkt; ++c) {
        const float* ctr = centroids + c * dim;
        float dist = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float diff = query[j] - ctr[j];
            dist += diff * diff;
        }
        out_dist[c] = std::sqrt(dist);
    }
}

// (d - mean) / scale
void standardize_distances(std::vector<float>& dist,
                           const std::vector<float>& mean,
                           const std::vector<float>& scale) {
    size_t n = dist.size();
    if (mean.size() != n || scale.size() != n) {
        throw std::runtime_error("Scaler dimension mismatch.");
    }
    for (size_t i = 0; i < n; ++i) {
        float s = scale[i];
        if (s == 0.0f) s = 1.0f;
        dist[i] = (dist[i] - mean[i]) / s;
    }
}

// L2 距离（平方），做 ranking 就够
float l2_sq(const float* a, const float* b, size_t dim) {
    float d = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
        float diff = a[j] - b[j];
        d += diff * diff;
    }
    return d;
}

// IP
float ip(const float* a, const float* b, size_t dim) {
    float s = 0.0f;
    for (size_t j = 0; j < dim; ++j) {
        s += a[j] * b[j];
    }
    return s;
}

// =========================== inverted list 结构 ===========================

struct Bucket {
    std::vector<int>   ids;   // global ids
    std::vector<float> data;  // ids.size() * dim
};

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 1;
    }
    omp_set_num_threads(args.num_threads);

    try {
        // ---------- 0. 路径 & 前缀 ----------
        std::string base = args.artifacts_dir;
        if (!base.empty() && base.back() != '/' && base.back() != '\\') {
            base += "/";
        }
        std::string prefix = base + args.prefix;

        std::cout << "Dataset      : " << args.dataset << "\n";
        std::cout << "Artifacts dir: " << base << "\n";
        std::cout << "Prefix       : " << prefix << "\n";
        std::cout << "Metric       : " << args.metric << "\n";
        std::cout << "K            : " << args.k << "\n";

        // ---------- 1. 载入 Python 端 artifacts ----------
        size_t n_bkt, dim_cent;
        std::vector<float> centroids =
            load_npy_float2d(prefix + "_centroids.npy", n_bkt, dim_cent);
        std::cout << "Loaded centroids: " << n_bkt << " x " << dim_cent << "\n";

        size_t n_data, n_mul;
        std::vector<int> data_2_bkt =
            load_npy_int2d(prefix + "_data_2_bkt.npy", n_data, n_mul);
        std::cout << "Loaded data_2_bkt: " << n_data << " x " << n_mul << "\n";

        size_t n_d2, dim;
        std::vector<float> x_d =
            load_npy_float2d(prefix + "_x_d.npy", n_d2, dim);
        if (n_d2 != n_data) {
            throw std::runtime_error("x_d.npy and data_2_bkt.npy mismatch in N.");
        }
        if (dim_cent != dim) {
            throw std::runtime_error("centroids dim and x_d dim mismatch.");
        }
        std::cout << "Loaded x_d: " << n_d2 << " x " << dim << "\n";

        size_t n_mean, n_scale;
        std::vector<float> scaler_mean  =
            load_npy_float1d(prefix + "_scaler_mean.npy", n_mean);
        std::vector<float> scaler_scale =
            load_npy_float1d(prefix + "_scaler_scale.npy", n_scale);
        if (n_mean != n_bkt || n_scale != n_bkt) {
            throw std::runtime_error("Scaler length must equal n_bkt.");
        }
        std::cout << "Loaded scaler parameters, len = " << n_mean << "\n";

        // TorchScript 模型（CPU）
        std::string model_path = prefix + "_mlp_2_input.pt";
        std::cout << "Loading TorchScript model from: " << model_path << "\n";
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval();
        torch::NoGradGuard no_grad;
        torch::Device device(torch::kCPU);

        // ---------- 2. 载入 query + groundtruth ----------
        std::string ds_dir = args.data_path + "/" + args.dataset;
        std::string q_path = ds_dir + "/" + args.dataset + "_query.fvecs";
        std::string gt_path = ds_dir + "/" + args.dataset + "_groundtruth.ivecs";

        size_t n_q, d_q;
        std::vector<float> x_q = read_fvecs(q_path, n_q, d_q);
        if (d_q != dim) {
            throw std::runtime_error("query dim != base dim.");
        }
        std::cout << "Loaded queries: " << n_q << " x " << d_q << "\n";

        size_t n_gt, d_gt;
        std::vector<int> gt_ids = read_ivecs(gt_path, n_gt, d_gt);
        if (n_gt != n_q) {
            throw std::runtime_error("groundtruth and queries count mismatch.");
        }
        if (args.k > (int)d_gt) {
            std::cerr << "Warning: requested k > groundtruth dim, clamp to " << d_gt << "\n";
            args.k = (int)d_gt;
        }

        bool is_ip = (args.metric == "inner_product" ||
                      args.metric == "IP" ||
                      args.metric == "ip");

        // ---------- 3. 构建 inverted list 布局 ----------
        std::cout << "Building inverted lists (buckets) ..." << std::endl;
        std::vector<std::vector<int>> bucket_ids(n_bkt);

        // 3.1 bucket_ids: bucket -> global ids
        for (size_t i = 0; i < n_data; ++i) {
            for (size_t j = 0; j < n_mul; ++j) {
                int b = data_2_bkt[i * n_mul + j];
                if (b < 0) continue;
                if (b >= (int)n_bkt) {
                    throw std::runtime_error("bucket id out of range.");
                }
                bucket_ids[b].push_back((int)i);
            }
        }
        for (size_t b = 0; b < n_bkt; ++b) {
            auto& ids = bucket_ids[b];
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        }

        // 3.2 为每个 bucket 构建连续 data
        std::vector<Bucket> buckets(n_bkt);
        for (size_t b = 0; b < n_bkt; ++b) {
            auto& ids = bucket_ids[b];
            Bucket& B = buckets[b];
            if (ids.empty()) continue;
            B.ids = ids;
            B.data.resize((size_t)ids.size() * dim);
            for (size_t i = 0; i < ids.size(); ++i) {
                int gid = ids[i];
                std::memcpy(
                    B.data.data() + i * dim,
                    x_d.data() + (size_t)gid * dim,
                    dim * sizeof(float)
                );
            }
        }
        std::cout << "Inverted lists built.\n";

        // 4. end-to-end：外层 threshold，内层 query
        std::vector<float> dist_to_centroids(n_bkt);

        std::cout << "Start end-to-end search (outer loop = threshold, inner = queries)\n";
        std::cout << "Threshold range: [" << args.t_min << ", "
                  << args.t_max << "] step " << args.t_step << "\n\n";

        for (float thr = args.t_min; thr <= args.t_max + 1e-6f; thr += args.t_step) {
            double sum_recall  = 0.0;
            double sum_nprobe  = 0.0;
            double sum_cmp     = 0.0;
            double total_time  = 0.0;

            std::cout << "=== Threshold = " << thr << " ===\n";

            for (size_t qi = 0; qi < n_q; ++qi) {
                const float* q = x_q.data() + qi * dim;

                auto t0 = std::chrono::high_resolution_clock::now();

                // 4.1 计算 query->centroids 距离 + 标准化
                compute_l2_to_centroids(q, centroids.data(), n_bkt, dim, dist_to_centroids);
                standardize_distances(dist_to_centroids, scaler_mean, scaler_scale);

                // 4.2 MLP 编码：scores = model(dist, q)
                torch::Tensor t_dist = torch::from_blob(
                    dist_to_centroids.data(), {1, (long)n_bkt}, torch::kFloat32
                ).clone(); // clone 避免后面被覆盖

                torch::Tensor t_vec = torch::from_blob(
                    (void*)q, {1, (long)dim}, torch::kFloat32
                ).clone();

                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(t_dist);
                inputs.push_back(t_vec);

                at::Tensor out_scores = model.forward(inputs).toTensor();  // [1, n_bkt]
                out_scores = out_scores.to(torch::kCPU).to(torch::kFloat32);
                auto acc = out_scores.accessor<float, 2>(); // [1][b]

                // 4.3 选出 score >= threshold 的 buckets（如果为空，则 fallback 到 argmax）
                std::vector<int> probed_buckets;
                probed_buckets.reserve(n_bkt);
                for (size_t b = 0; b < n_bkt; ++b) {
                    float s = acc[0][b];
                    if (s >= thr) {
                        probed_buckets.push_back((int)b);
                    }
                }
                if (probed_buckets.empty()) {
                    int best_b = 0;
                    float best_s = acc[0][0];
                    for (size_t b = 1; b < n_bkt; ++b) {
                        if (acc[0][b] > best_s) {
                            best_s = acc[0][b];
                            best_b = (int)b;
                        }
                    }
                    probed_buckets.push_back(best_b);
                }

                long cmp_for_query = 0;  // DCO: 向量级 distance computation 次数
                std::vector<std::pair<float, int>> candidates;

                // 4.4 在这些 buckets 里做精确 DCO
                for (int b : probed_buckets) {
                    const Bucket& B = buckets[b];
                    if (B.ids.empty()) continue;

                    size_t sz = B.ids.size();
                    cmp_for_query += (long)sz;

                    const float* data_ptr = B.data.data();
                    for (size_t i = 0; i < sz; ++i) {
                        const float* v = data_ptr + i * dim;
                        float score;
                        if (is_ip) {
                            // 对 IP：越大越好，统一用 -IP 让 "越小越好"
                            score = -ip(q, v, dim);
                        } else {
                            // L2：越小越好
                            score = l2_sq(q, v, dim);
                        }
                        int gid = B.ids[i];
                        candidates.emplace_back(score, gid);
                    }
                }

                // 4.5 对所有 candidates 取 top-k
                std::unordered_set<int> cand_ids;
                cand_ids.reserve(args.k * 4);

                if (!candidates.empty()) {
                    size_t topk = (candidates.size() < (size_t)args.k)
                                  ? candidates.size()
                                  : (size_t)args.k;
                    std::nth_element(
                        candidates.begin(),
                        candidates.begin() + topk,
                        candidates.end(),
                        [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                            return a.first < b.first; // 越小越好
                        }
                    );
                    for (size_t i = 0; i < topk; ++i) {
                        cand_ids.insert(candidates[i].second);
                    }
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t1 - t0).count();

                // 4.6 计算 Recall@k
                const int* gt_ptr = gt_ids.data() + qi * d_gt;
                int hit = 0;
                for (int j = 0; j < args.k; ++j) {
                    int g = gt_ptr[j];
                    if (cand_ids.find(g) != cand_ids.end()) {
                        hit++;
                    }
                }
                double recall_q = (double)hit / (double)args.k;

                sum_recall += recall_q;
                sum_nprobe += (double)probed_buckets.size();
                sum_cmp    += (double)cmp_for_query;
                total_time += elapsed;
            }

            double avg_recall = sum_recall / (double)n_q;
            double avg_nprobe = sum_nprobe / (double)n_q;
            double avg_cmp    = sum_cmp    / (double)n_q;
            double avg_time   = total_time / (double)n_q;
            double qps        = (double)n_q / total_time;

            std::cout << "Threshold    : " << thr << "\n";
            std::cout << "avg_recall   : " << avg_recall << "\n";
            std::cout << "avg_nprobe   : " << avg_nprobe << "\n";
            std::cout << "avg_cmp      : " << avg_cmp << "\n";
            std::cout << "avg_time(q)  : " << avg_time << " s\n";
            std::cout << "QPS          : " << qps << " q/s\n";
            std::cout << "----------------------------------------\n";
        }

        std::cout << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
