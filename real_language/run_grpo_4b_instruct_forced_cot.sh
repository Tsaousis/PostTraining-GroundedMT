#!/bin/bash
# forced-CoT chrF-GRPO, Qwen3-4B (instruct, thinking on), clean prompt, 1,998 stratified MIK prompts, GPU1
cd swells && source venv/bin/activate
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=real_language/data/runs/grpo_forced; mkdir -p $OUT
MASTER_PORT=12355 CUDA_VISIBLE_DEVICES=1 SWELLS_STRICT_TAG=1 SWELLS_MIN_REASON_CHARS=200 SWELLS_LOG_FULL=1 SWELLS_REWARD_LOG=$OUT/rewards.jsonl \
python grpo_train.py --model Qwen/Qwen3-4B --train_file real_language/data/train/b2s_train_mik_cot_forced_sub2k.jsonl --output_dir $OUT --lora \
  --epochs 1 --num_generations 8 --batch_size 1 --grad_accum 8 --max_prompt_length 7168 --max_completion_length 2048 \
  --lr 1e-5 --beta 0.0 --epsilon_high 0.28 --vllm_mode colocate --vllm_gpu_util 0.45 --vllm_server_port 8014 > $OUT.log 2>&1 \
 && CUDA_VISIBLE_DEVICES=1 python merge_lora.py --base Qwen/Qwen3-4B --adapter $OUT/final --out $OUT/merged >> $OUT.log 2>&1 \
