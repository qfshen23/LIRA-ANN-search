export LIBTORCH=~/workspace/libs/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:/usr/local/lib:$LD_LIBRARY_PATH
# -ltorch -lc10 \
g++ -std=gnu++17 search.cpp \
  -I${LIBTORCH}/include \
  -I${LIBTORCH}/include/torch/csrc/api/include \
  -I/usr/local/include \
  -L${LIBTORCH}/lib \
  -L/usr/local/lib \
  -lcnpy -lz -lc10 -ltorch -ltorch_cpu \
  -fopenmp -O3 \
  -o search

# may change
datasets=("sift" "gist" "tiny5m" "sift10m" "deep10m" "bigann10m")

# 每个数据集下可以配置多组 “n_bkt|metric|re_ratio”，用空格分隔多组
declare -A dataset_param_sets
dataset_param_sets["sift"]="64|L2|0.03 256|L2|0.03 512|L2|0.03 1024|L2|0.03"
dataset_param_sets["gist"]="64|L2|0.03 256|L2|0.03 512|L2|0.03 1024|L2|0.03"
dataset_param_sets["tiny5m"]="64|L2|0.03 256|L2|0.03 512|L2|0.03 1024|L2|0.03"
dataset_param_sets["sift10m"]="256|L2|0.03 2048|L2|0.03"
dataset_param_sets["deep10m"]="256|L2|0.03 2048|L2|0.03"
dataset_param_sets["bigann10m"]="256|L2|0.03 2048|L2|0.03"

# fixed
k=10
t_min=0.02
t_max=0.80
t_step=0.02
threads=32

for dataset in "${datasets[@]}"; do
  param_sets=${dataset_param_sets[$dataset]}

  if [[ -z "${param_sets}" ]]; then
    echo "[WARN] dataset ${dataset} 没有配置参数集合，跳过"
    continue
  fi

  for param_set in ${param_sets}; do
    IFS='|' read -r n_bkt metric re_ratio <<< "${param_set}"

    ./search \
      --dataset ${dataset} \
      --artifacts_dir /data/tmp/lira/${dataset}/ML_kmeans_RE_FLAT/ \
      --prefix ${dataset}-k=${k}-ML_kmeans=${n_bkt}_FLAT_Metric=${metric}_ReType=model_ReRatio=${re_ratio} \
      --k ${k} \
      --metric ${metric} \
      --t_min ${t_min} --t_max ${t_max} --t_step ${t_step} --num_threads ${threads}
  done
done
