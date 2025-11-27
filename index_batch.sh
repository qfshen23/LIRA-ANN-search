#!/usr/bin/env bash

source venv/bin/activate

run_cmd() {
  echo "Running: $*"
  "$@"
  if [ $? -ne 0 ]; then
    echo "Command FAILED: $*" >&2
  else
    echo "Command SUCCEEDED: $*"
  fi
}

run_cmd python3 index.py --dataset sift --n_bkt 64 --k 10
run_cmd python3 index.py --dataset sift --n_bkt 256 --k 10
run_cmd python3 index.py --dataset sift --n_bkt 512 --k 10
run_cmd python3 index.py --dataset sift --n_bkt 1024 --k 10

run_cmd python3 index.py --dataset gist --n_bkt 64 --k 10
run_cmd python3 index.py --dataset gist --n_bkt 256 --k 10
run_cmd python3 index.py --dataset gist --n_bkt 512 --k 10
run_cmd python3 index.py --dataset gist --n_bkt 1024 --k 10

run_cmd python3 index.py --dataset tiny5m --n_bkt 64 --k 10
run_cmd python3 index.py --dataset tiny5m --n_bkt 256 --k 10
run_cmd python3 index.py --dataset tiny5m --n_bkt 2048 --k 10

run_cmd python3 index.py --dataset sift10m --n_bkt 256 --k 10
run_cmd python3 index.py --dataset sift10m --n_bkt 2048 --k 10

run_cmd python3 index.py --dataset deep10m --n_bkt 256 --k 10
run_cmd python3 index.py --dataset deep10m --n_bkt 2048 --k 10

run_cmd python3 index.py --dataset bigann10m --n_bkt 256 --k 10
run_cmd python3 index.py --dataset bigann10m --n_bkt 2048 --k 10
