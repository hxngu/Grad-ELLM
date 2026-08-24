#!/usr/bin/env bash
set -euo pipefail

python evaluate.py \
  --method random \
  --batch_size 1 \
  --evaluation soft_ns_nc \
  --evaluation_mode per_token \
  --scaling_method alpha \
  --device 0 \
  --model llama3.1-8b-instruct \
  --dataset BoolQ
