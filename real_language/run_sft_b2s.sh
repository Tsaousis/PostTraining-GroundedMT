#!/bin/bash
cd swells && source venv/bin/activate
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 python sft_train.py --model Qwen/Qwen3-4B-Base --train_file real_language/data/train/b2s_train_mizo_ilokano_kimbundu.jsonl --output_dir real_language/data/runs/sft_mizo_ilo_kim --lora --epochs 3 --per_device_bs 1 --grad_accum 16 --lr 1e-4 --max_len 10240 > real_language/data/logs_sft.log 2>&1 && echo OK_SFT >> real_language/data/status.txt || echo FAIL_SFT >> real_language/data/status.txt
