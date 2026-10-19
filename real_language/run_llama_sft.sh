#!/bin/bash
# GPU1: Llama-3.2-3B-Instruct SFT on the same 5,982 MIK prompts as Qwen (no retrieved examples) -> merge -> Hus eval (old contexts)
cd swells && source venv/bin/activate
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=unsloth/Llama-3.2-3B-Instruct; TRAIN=real_language/data/train/b2s_train_mizo_ilokano_kimbundu.jsonl; G=1
SFT=real_language/data/runs/llama_sft_mik; mkdir -p $SFT
CUDA_VISIBLE_DEVICES=$G python sft_train.py --model $M --train_file $TRAIN --output_dir $SFT --lora --epochs 3 --per_device_bs 1 --grad_accum 16 --lr 1e-4 --max_len 10240 --fail_on_skip > real_language/data/logs_llama_sft.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=$G python merge_lora.py --base $M --adapter $SFT/final --out $SFT/merged >> real_language/data/logs_llama_sft.log 2>&1 && echo OK_llama_sft >> real_language/data/status.txt || { echo FAIL_llama_sft >> real_language/data/status.txt; exit 1; }
CUDA_VISIBLE_DEVICES=$G python real_language/eval_b2s.py --model $SFT/merged --tag llama3b-sft --suffix > real_language/data/logs_eval_llama3b-sft.log 2>&1 && echo OK_EVAL_llama3b-sft >> real_language/data/status.txt || echo FAIL_EVAL_llama3b-sft >> real_language/data/status.txt
echo LLAMA_SFT_DONE >> real_language/data/status.txt
