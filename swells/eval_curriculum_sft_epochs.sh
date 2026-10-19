#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-runs/sft_curriculum}
TEST_ROOT=${TEST_ROOT:-./swells_min/SWELLS_Dataset/Test/Test_CoT}
OUT=${OUT:-eval/sft_curriculum_epochs}

until [[ -s "$ROOT/final/merged/config.json" ]]; do
  sleep 300
done

mapfile -t checkpoints < <(
  find "$ROOT/final" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
)
if [[ ${#checkpoints[@]} -ne 4 ]]; then
  echo "expected 4 final-stage checkpoints, found ${#checkpoints[@]}" >&2
  exit 1
fi

tests=(
  "$TEST_ROOT/CoT_full_art_eng_test_grammar_excerpts.jsonl"
  "$TEST_ROOT/CoT_full_eng_art_test_grammar_excerpts.jsonl"
)

for index in "${!checkpoints[@]}"; do
  epoch=$((index + 1))
  merged="$ROOT/final/epoch_${epoch}_merged"
  results="$OUT/epoch_$epoch"
  if [[ ! -s "$merged/config.json" ]]; then
    python merge_lora.py \
      --base "$ROOT/course_7/merged" \
      --adapter "${checkpoints[$index]}" \
      --out "$merged"
  fi
  if [[ ! -s "$results/summary.json" ]]; then
    python evaluate.py \
      --model "$merged" \
      --test_files "${tests[@]}" \
      --output_dir "$results" \
      --per_partition 10 \
      --tp 4 \
      --max_model_len 16384 \
      --max_tokens 2048
  fi
done
