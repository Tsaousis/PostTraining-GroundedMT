#!/bin/bash
# Qwen3-8B instruct, thinking on, explicit-CoT chrF-GRPO on 1,998 MIK real-language prompts (SWELLS format), LoRA, 1 epoch.
source venv/bin/activate; cd swells
cp swells/grpo_train.py swells/swells_reward.py swells/merge_lora.py swells/
OUT=real_language/data/runs/grpo8b_real_cot; mkdir -p $OUT; MODEL=Qwen/Qwen3-8B
CUDA_VISIBLE_DEVICES=1 nohup trl vllm-serve --model $MODEL --port 8018 --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 10240 > $OUT/vllm.log 2>&1 &
for i in $(seq 1 60); do curl -s localhost:8018/health >/dev/null && break; sleep 10; done
CKPT=$(find $OUT -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1); RES=(); [ -n "$CKPT" ] && RES=(--resume_from_checkpoint "$CKPT")
CUDA_VISIBLE_DEVICES=0 SWELLS_STRICT_TAG=1 SWELLS_LOG_FULL=1 SWELLS_REWARD_LOG=$OUT/rewards.jsonl \
python grpo_train.py --model $MODEL --train_file real_language/data/train/b2s_train_mik_sub2000_swellsfmt.jsonl --output_dir $OUT --lora --epochs 1 \
  --num_generations 8 --batch_size 1 --grad_accum 8 --max_prompt_length 6144 --max_completion_length 2048 \
  --lr 1e-5 --beta 0.0 --epsilon_high 0.28 --vllm_mode server --vllm_server_port 8018 "${RES[@]}" > $OUT/train.log 2>&1
RC=$?; pkill -f "vllm-serve --model $MODEL" ; sleep 20
