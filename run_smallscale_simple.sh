#!/usr/bin/env bash

run_cmd() {
  echo "Running: $*"
  "$@"
  if [ $? -ne 0 ]; then
    echo "Command FAILED: $*" >&2
  else
    echo "Command SUCCEEDED: $*"
  fi
}

# L2 Datasets
run_cmd python3 LIRA_smallscale.py --dataset sift --n_bkt 1024 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset sift --n_bkt 1024 --k 1 --dis_metric L2

# run_cmd python3 LIRA_smallscale.py --dataset gist --n_bkt 512 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset gist --n_bkt 512 --k 1 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset gist --n_bkt 1024 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset gist --n_bkt 1024 --k 1 --dis_metric L2

# run_cmd python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 512 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset tiny5m --n_bkt 512 --k 1 --dis_metric L2

# run_cmd python3 LIRA_smallscale.py --dataset sift10m --n_bkt 256 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset sift10m --n_bkt 256 --k 1 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset sift10m --n_bkt 512 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset sift10m --n_bkt 512 --k 1 --dis_metric L2

# run_cmd python3 LIRA_smallscale.py --dataset deep10m --n_bkt 256 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset deep10m --n_bkt 256 --k 1 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset deep10m --n_bkt 512 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset deep10m --n_bkt 512 --k 1 --dis_metric L2

# run_cmd python3 LIRA_smallscale.py --dataset bigann10m --n_bkt 256 --k 10 --dis_metric L2     
# run_cmd python3 LIRA_smallscale.py --dataset bigann10m --n_bkt 256 --k 1 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset bigann10m --n_bkt 512 --k 10 --dis_metric L2
# run_cmd python3 LIRA_smallscale.py --dataset bigann10m --n_bkt 512 --k 1 --dis_metric L2

# IP Datasets
# run_cmd python3 LIRA_smallscale.py --dataset openai1536 --n_bkt 256 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai1536 --n_bkt 256 --k 1 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai1536 --n_bkt 512 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai1536 --n_bkt 512 --k 1 --dis_metric ip

# run_cmd python3 LIRA_smallscale.py --dataset openai3072 --n_bkt 256 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai3072 --n_bkt 256 --k 1 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai3072 --n_bkt 512 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset openai3072 --n_bkt 512 --k 1 --dis_metric ip

# run_cmd python3 LIRA_smallscale.py --dataset glove2m_normalized --n_bkt 256 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset glove2m_normalized --n_bkt 256 --k 1 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset glove2m_normalized --n_bkt 512 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset glove2m_normalized --n_bkt 512 --k 1 --dis_metric ip

# run_cmd python3 LIRA_smallscale.py --dataset word2vec_normalized --n_bkt 256 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset word2vec_normalized --n_bkt 256 --k 1 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset word2vec_normalized --n_bkt 512 --k 10 --dis_metric ip
# run_cmd python3 LIRA_smallscale.py --dataset word2vec_normalized --n_bkt 512 --k 1 --dis_metric ip