#!/usr/bin/env bash
# Annealing grammar-dropout SFT (Zhu et al. 2025) on the frozen SWELLS curriculum.
# Same stages/hyper-parameters as run_curriculum_sft.sh (effective batch 16, lr 1e-4,
# LoRA merged between stages); stage files come from build_dropout_schedule.py and
# are consumed in order (--no_shuffle) so the global dropout schedule is respected.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
DATA=${DATA:-${SWELLS_SCHEDULE:-data/swells_curriculum/sft}_dropout/sft_e4/cot}
OUT=${OUT:-runs/sft_dropout}
MAXLEN=${MAXLEN:-16384}
GRAD_ACCUM=${GRAD_ACCUM:-16}

mkdir -p "$OUT"
current_model=$MODEL
for stage_name in course_1 course_2 course_3 course_4 course_5 course_6 course_7 final_ep1 final_ep2 final_ep3 final_ep4; do
  stage="$OUT/$stage_name"
  if [[ ! -s "$stage/merged/config.json" ]]; then
    python3 sft_train.py \
      --model "$current_model" \
      --train_file "$DATA/$stage_name.jsonl" \
      --output_dir "$stage" \
      --lora \
      --epochs 1 \
      --per_device_bs 1 \
      --grad_accum "$GRAD_ACCUM" \
      --lr 1e-4 \
      --max_len "$MAXLEN" \
      --no_shuffle \
      --fail_on_skip
  fi
  prev=$current_model
  current_model="$stage/merged"
  # keep disk bounded: only the latest merged checkpoint plus the final one are needed
  if [[ "$prev" == "$OUT"/* && "$prev" != */final_ep*/merged ]]; then rm -rf "$prev"; fi
done
echo "$current_model"
