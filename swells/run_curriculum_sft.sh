#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
DATA=${DATA:-${SWELLS_SCHEDULE:-data/swells_curriculum/sft}}
OUT=${OUT:-runs/sft_curriculum}
MAXLEN=${MAXLEN:-16384}
GPUS=${GPUS:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
DISABLE_THINKING=${DISABLE_THINKING:-0}

mkdir -p "$OUT"
current_model=$MODEL
thinking_args=()
if [[ "$DISABLE_THINKING" == 1 ]]; then
  thinking_args=(--disable_thinking)
fi

for course in 1 2 3 4 5 6 7; do
  stage="$OUT/course_$course"
  if [[ ! -s "$stage/merged/config.json" ]]; then
    torchrun --nproc_per_node="$GPUS" sft_train.py \
      --model "$current_model" \
      --train_file "$DATA/course_$course.jsonl" \
      --output_dir "$stage" \
      --lora \
      --epochs 1 \
      --per_device_bs 1 \
      --grad_accum "$GRAD_ACCUM" \
      --lr 1e-4 \
      --max_len "$MAXLEN" \
      "${thinking_args[@]}"
  fi
  current_model="$stage/merged"
done

stage="$OUT/final"
if [[ ! -s "$stage/merged/config.json" ]]; then
  torchrun --nproc_per_node="$GPUS" sft_train.py \
    --model "$current_model" \
    --train_file "$DATA/final_1408.jsonl" \
    --output_dir "$stage" \
    --lora \
    --epochs 4 \
    --per_device_bs 1 \
    --grad_accum "$GRAD_ACCUM" \
    --lr 1e-4 \
    --max_len "$MAXLEN" \
    "${thinking_args[@]}"
fi

echo "$stage/merged"
