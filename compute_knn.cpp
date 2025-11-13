#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <sys/stat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>

// Read .fvecs/.bvecs format files
template<typename T>
std::vector<float> read_xvecs(const std::string& filename, int& n, int& dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    
    // Read first vector to get dimension
    file.read((char*)&dim, sizeof(int));
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    n = file_size / ((dim + 1) * sizeof(T));
    std::cout << "Reading " << n << " vectors of dimension " << dim << std::endl;
    
    std::vector<float> data(n * dim);
    std::vector<T> buffer(dim);
    
    for (int i = 0; i < n; i++) {
        int d;
        file.read((char*)&d, sizeof(int));
        if (d != dim) {
            std::cerr << "Error: Inconsistent dimension at vector " << i << std::endl;
            exit(1);
        }
        file.read((char*)buffer.data(), dim * sizeof(T));
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = static_cast<float>(buffer[j]);
        }
        
        if ((i + 1) % 100000 == 0) {
            std::cout << "  Read " << (i + 1) << " vectors..." << std::endl;
        }
    }
    
    file.close();
    return data;
}

// Save results to binary format (compatible with numpy)
void save_to_npy(const std::string& filename, const std::vector<int>& data, int rows, int cols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        exit(1);
    }
    
    // Simple binary format (can be read with np.fromfile and reshape)
    // For full .npy format, we'd need to write numpy header, but this is simpler
    file.write((char*)data.data(), data.size() * sizeof(int));
    file.close();
    
    std::cout << "Saved KNN results to: " << filename << std::endl;
    std::cout << "  Shape: (" << rows << ", " << cols << ")" << std::endl;
    std::cout << "  To load in Python: np.fromfile('" << filename << "', dtype=np.int32).reshape(" 
              << rows << ", " << cols << ")" << std::endl;
}

// Create directory if not exists
void create_directory(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        std::string cmd = "mkdir -p " + path;
        int ret = system(cmd.c_str());
        (void)ret;  // Suppress unused result warning
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <dataset_name> <data_path> <k> [nprobe] [n_threads]" << std::endl;
        std::cout << "Example: " << argv[0] << " sift /data/vector_datasets 10 64 24" << std::endl;
        std::cout << "  nprobe: number of clusters to probe (default: auto, 0=exact search)" << std::endl;
        std::cout << "  n_threads: number of OpenMP threads (default: all cores)" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommended nprobe values:" << std::endl;
        std::cout << "  Fast (lower accuracy):    nprobe = 16-32" << std::endl;
        std::cout << "  Balanced (recommended):   nprobe = 64-128" << std::endl;
        std::cout << "  High accuracy:            nprobe = 256-512" << std::endl;
        std::cout << "  Exact search:             nprobe = 0" << std::endl;
        return 1;
    }
    
    std::string dataset_name = argv[1];
    std::string data_path = argv[2];
    int k = std::stoi(argv[3]);
    int nprobe = (argc > 4) ? std::stoi(argv[4]) : -1;  // -1 means auto
    int n_threads = (argc > 5) ? std::stoi(argv[5]) : omp_get_max_threads();
    
    omp_set_num_threads(n_threads);
    
    std::cout << "=== FAISS KNN Computation ===" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "Threads: " << n_threads << std::endl;
    
    // Construct file paths
    std::string dataset_dir = data_path + "/" + dataset_name;
    std::string base_file = dataset_dir + "/" + dataset_name + "_base.fvecs";
    
    // Try .bvecs if .fvecs doesn't exist
    struct stat buffer;
    bool is_bvecs = false;
    if (stat(base_file.c_str(), &buffer) != 0) {
        base_file = dataset_dir + "/" + dataset_name + "_base.bvecs";
        is_bvecs = true;
        if (stat(base_file.c_str(), &buffer) != 0) {
            std::cerr << "Error: Cannot find base file for dataset " << dataset_name << std::endl;
            return 1;
        }
    }
    
    // Read data
    int n, dim;
    std::vector<float> data;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    if (is_bvecs) {
        std::cout << "Reading .bvecs file..." << std::endl;
        data = read_xvecs<uint8_t>(base_file, n, dim);
    } else {
        std::cout << "Reading .fvecs file..." << std::endl;
        data = read_xvecs<float>(base_file, n, dim);
    }
    
    auto t_read = std::chrono::high_resolution_clock::now();
    double read_time = std::chrono::duration<double>(t_read - t_start).count();
    std::cout << "Read time: " << read_time << "s" << std::endl;
    std::cout << std::endl;
    
    // Build index
    std::cout << "Building FAISS index..." << std::endl;
    faiss::Index* index = nullptr;
    faiss::IndexIVFFlat* ivf_index = nullptr;
    faiss::IndexFlatL2* quantizer = nullptr;  // Keep quantizer alive
    int actual_nprobe = 0;
    bool is_approximate = (nprobe != 0);  // 0 means exact search
    
    if (is_approximate) {
        // Use IVF for approximate search (RECOMMENDED for speed)
        // Calculate optimal number of clusters
        int n_list;
        if (n < 50000) {
            n_list = std::min(static_cast<int>(std::sqrt(n)), 256);
        } else if (n < 1000000) {
            n_list = std::min(static_cast<int>(std::sqrt(n)), 1024);
        } else {
            n_list = std::min(static_cast<int>(std::sqrt(n)), 4096);
        }
        
        std::cout << "Using IVF index with " << n_list << " clusters" << std::endl;
        
        // Create quantizer dynamically to keep it alive
        quantizer = new faiss::IndexFlatL2(dim);
        ivf_index = new faiss::IndexIVFFlat(quantizer, dim, n_list);
        ivf_index->own_fields = true;  // Let IVF index own the quantizer
        index = ivf_index;
        
        std::cout << "Training index..." << std::endl;
        auto t_train_start = std::chrono::high_resolution_clock::now();
        index->train(n, data.data());
        auto t_train_end = std::chrono::high_resolution_clock::now();
        double train_time = std::chrono::duration<double>(t_train_end - t_train_start).count();
        std::cout << "Training time: " << train_time << "s" << std::endl;
        
        std::cout << "Adding vectors..." << std::endl;
        index->add(n, data.data());
        
        // Set nprobe (auto-calculate if not specified)
        if (nprobe < 0) {
            // Auto: balance between speed and accuracy
            if (n < 100000) {
                actual_nprobe = std::min(std::max(n_list / 4, 16), 64);
            } else {
                actual_nprobe = std::min(std::max(n_list / 8, 32), 128);
            }
        } else {
            actual_nprobe = nprobe;
        }
        
        ivf_index->nprobe = actual_nprobe;
        
        std::cout << "Set nprobe = " << actual_nprobe << " (out of " << n_list << " clusters)" << std::endl;
        std::cout << "  Probe ratio: " << (100.0 * actual_nprobe / n_list) << "%" << std::endl;
        std::cout << "  Expected accuracy: ~" << (95 + 5.0 * actual_nprobe / n_list) << "%" << std::endl;
        std::cout << "Method: Approximate IVF search" << std::endl;
        
    } else {
        // Use exact search (SLOW for large datasets)
        std::cout << "Using exact FLAT index (this may be slow!)" << std::endl;
        std::cout << "Tip: Use nprobe > 0 for much faster approximate search" << std::endl;
        index = new faiss::IndexFlatL2(dim);
        index->add(n, data.data());
        std::cout << "Method: Exact FLAT search" << std::endl;
    }
    
    std::cout << std::endl;
    
    auto t_build = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t_build - t_read).count();
    std::cout << "Index build time: " << build_time << "s" << std::endl;
    std::cout << std::endl;
    
    // Search KNN
    std::cout << "Computing self-KNN (k=" << k << ")..." << std::endl;
    
    // Allocate result arrays
    std::vector<float> distances(n * (k + 1));  // +1 to include self
    std::vector<faiss::idx_t> labels(n * (k + 1));
    
    // Batch search for better performance
    int batch_size = std::min(10000, n);
    std::cout << "Searching in batches of " << batch_size << "..." << std::endl;
    
    auto t_search_start = std::chrono::high_resolution_clock::now();
    
    for (int start = 0; start < n; start += batch_size) {
        int end = std::min(start + batch_size, n);
        int batch_n = end - start;
        
        index->search(batch_n, data.data() + start * dim, k + 1,
                     distances.data() + start * (k + 1),
                     labels.data() + start * (k + 1));
        
        if ((start + batch_size) % 100000 == 0 || end == n) {
            std::cout << "  Searched " << end << " / " << n << " vectors" << std::endl;
        }
    }
    
    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_time = std::chrono::duration<double>(t_search_end - t_search_start).count();
    std::cout << "Search time: " << search_time << "s" << std::endl;
    std::cout << "Average: " << (search_time / n * 1000) << " ms/query" << std::endl;
    std::cout << std::endl;
    
    // Remove self from results (first neighbor is usually itself)
    std::cout << "Removing self from KNN results..." << std::endl;
    std::vector<int> knn_results(n * k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            knn_results[i * k + j] = static_cast<int>(labels[i * (k + 1) + j + 1]);
        }
    }
    
    // Save results
    std::string cache_dir = dataset_dir + "/knn_cache";
    create_directory(cache_dir);
    
    // Include nprobe in filename for approximate search
    std::string suffix = is_approximate ? ("_ivf_nprobe" + std::to_string(actual_nprobe)) : "";
    std::string output_file = cache_dir + "/" + dataset_name + 
                              "-data_self_knn" + std::to_string(k) + 
                              "-n" + std::to_string(n) + suffix + ".bin";
    
    save_to_npy(output_file, knn_results, n, k);
    
    // Also save metadata
    std::string meta_file = output_file + ".meta";
    std::ofstream meta(meta_file);
    meta << "dataset: " << dataset_name << std::endl;
    meta << "n: " << n << std::endl;
    meta << "dim: " << dim << std::endl;
    meta << "k: " << k << std::endl;
    meta << "method: " << (is_approximate ? "ivf_approximate" : "flat_exact") << std::endl;
    if (is_approximate && ivf_index) {
        meta << "n_clusters: " << ivf_index->nlist << std::endl;
        meta << "nprobe: " << actual_nprobe << std::endl;
        meta << "probe_ratio: " << (100.0 * actual_nprobe / ivf_index->nlist) << "%" << std::endl;
    }
    meta << "read_time: " << read_time << "s" << std::endl;
    meta << "build_time: " << build_time << "s" << std::endl;
    meta << "search_time: " << search_time << "s" << std::endl;
    meta << "total_time: " << (read_time + build_time + search_time) << "s" << std::endl;
    meta.close();
    
    // Print summary
    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Total time: " << (read_time + build_time + search_time) << "s" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << std::endl;
    std::cout << "To load in Python:" << std::endl;
    std::cout << "  knn_data = np.fromfile('" << output_file << "', dtype=np.int32).reshape(" 
              << n << ", " << k << ")" << std::endl;
    
    // Clean up (quantizer will be deleted automatically by IVF index if own_fields=true)
    delete index;
    
    return 0;
}

