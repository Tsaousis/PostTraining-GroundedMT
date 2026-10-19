#!/usr/bin/env bash
# Grader-v2 reward pilot: Qwen3-4B-Base -> direct/no-CoT GRPO on ONE morphology course
# (course 4, 370 prompts, 1 epoch) with the rule-decomposed reward instead of chrF.
# Everything else (data, ordering, hyper-parameters) matches the chrF curriculum runs.
# Go/no-go for a full-curriculum v2 arm: rule component trajectory, EM diagnostic,
# and official partition-4 accuracy vs base and chrF-GRPO-from-base.
set -euo pipefail
cd .
source venv/bin/activate

RUN=${RUN:-grpo_v2_pilot_course4}
export MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
export OUT=runs/$RUN
export MODE=colocate
export GPU_UTIL=${GPU_UTIL:-0.35}
export REWARD=v2
export COURSES=${COURSES:-4}
export FINAL_EPOCHS=0

mkdir -p "$OUT"

SOURCE=./$OUT DEST=./a100-checkpoints/$RUN \
  nohup bash backup_a100_checkpoints.sh > "$OUT/backup.log" 2>&1 < /dev/null &

for attempt in $(seq 1 10); do
  if bash run_curriculum_grpo.sh >> "$OUT/train.log" 2>&1; then
    echo "V2_PILOT_DONE $(date -u)" >> "$OUT/train.log"
    break
  fi
  echo "runner exited with error (attempt $attempt), restarting in 60s" >> "$OUT/train.log"
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  sleep 60
done
