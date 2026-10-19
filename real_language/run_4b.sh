#!/bin/bash
cd real_language; source venv/bin/activate; export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 python qwen/eval_b2s.py --model Qwen/Qwen3-4B-Base --tag qwen3-4b-base --completion > logs_4b.log 2>&1 && echo OK_4B >> status.txt || echo FAIL_4B >> status.txt
