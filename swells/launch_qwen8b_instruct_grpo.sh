#!/usr/bin/env bash
# Qwen3-8B (official instruct) -> direct/no-CoT chrF-GRPO on the frozen SWELLS curriculum.
# 2x A100-40GB: trainer on GPU 0, vLLM generation server on GPU 1.
# Same data, ordering, schedule and hyper-parameters as the 4B runs; only MODEL and
# DISABLE_THINKING (Qwen3 non-thinking marker) differ. Idempotent: re-running resumes.
set -euo pipefail
cd .
source venv/bin/activate

RUN=${RUN:-grpo_curriculum_qwen8b_instruct}
export MODEL=${MODEL:-Qwen/Qwen3-8B}
export OUT=runs/$RUN
export MODE=server
export DISABLE_THINKING=1
export TRAIN_VISIBLE=0
export TRAIN_PROCESSES=1
export VLLM_VISIBLE=1
export VLLM_TP=1
export VLLM_DP=1
export FINAL_EPOCHS=${FINAL_EPOCHS:-2}

mkdir -p "$OUT"

# rolling checkpoint backup to the L4 and stable epoch-1 snapshot
if ! pgrep -f "backup_a100_checkpoints.sh" >/dev/null; then
  SOURCE=./$OUT DEST=~/a100-checkpoints/$RUN \
    nohup bash backup_a100_checkpoints.sh > "$OUT/backup.log" 2>&1 < /dev/null &
fi
if ! pgrep -f "capture_epoch1_ckpt.sh" >/dev/null; then
  RUN=./$OUT EPOCH_STEPS=1408 \
    nohup bash capture_epoch1_ckpt.sh > "$OUT/capture_epoch1.log" 2>&1 < /dev/null &
fi

# auto-restart the curriculum runner if it dies (each stage resumes from its last checkpoint)
for attempt in $(seq 1 20); do
  if bash run_curriculum_grpo.sh >> "$OUT/train.log" 2>&1; then
    echo "GRPO_8B_DONE $(date -u)" >> "$OUT/train.log"
    break
  fi
  echo "runner exited with error (attempt $attempt), restarting in 60s" >> "$OUT/train.log"
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  pkill -f 'trl vllm-serve' 2>/dev/null || true
  sleep 60
done
