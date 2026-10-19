#!/bin/bash
set -uo pipefail
cd real_language; source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python qwen/eval_b2s.py --model Qwen/Qwen3-4B-Base --tag qwen3-4b-base > logs_4b.log 2>&1 && echo OK_4B >> status.txt || echo FAIL_4B >> status.txt &
CUDA_VISIBLE_DEVICES=1 python qwen/eval_b2s.py --model Qwen/Qwen3-8B --tag qwen3-8b-instruct-nothink > logs_8b.log 2>&1 && echo OK_8B >> status.txt || echo FAIL_8B >> status.txt &
wait
