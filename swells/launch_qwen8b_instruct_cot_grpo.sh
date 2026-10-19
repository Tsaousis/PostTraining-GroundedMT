#!/usr/bin/env bash
# Qwen3-8B (official instruct) -> CoT chrF-GRPO on the frozen SWELLS curriculum.
# Identical data points, ordering, schedule, reward (chrF) and optimiser settings as
# launch_qwen8b_instruct_grpo.sh (the no-CoT control). Differences, by design:
#   * prompts are the SWELLS CoT prompts ("you may explain your chain of thoughts")
#   * Qwen3 thinking is left enabled, so rollouts reason before <translation>
#   * completion cap raised to MAX_COMPLETION tokens to fit the reasoning
# 2x A100-40GB: trainer on GPU 0, vLLM generation server on GPU 1. Idempotent: resumes.
set -euo pipefail
cd .
source venv/bin/activate

RUN=${RUN:-grpo_curriculum_qwen8b_instruct_cot}
export MODEL=${MODEL:-Qwen/Qwen3-8B}
export DATA=${SWELLS_SCHEDULE:-data/swells_curriculum/sft}
export OUT=runs/$RUN
export MODE=server
export DISABLE_THINKING=0
export REWARD=chrf
export MAX_COMPLETION=${MAX_COMPLETION:-1536}
export TRAIN_VISIBLE=0
export TRAIN_PROCESSES=1
export VLLM_VISIBLE=1
export VLLM_TP=1
export VLLM_DP=1
export FINAL_EPOCHS=${FINAL_EPOCHS:-2}

mkdir -p "$OUT"

if [[ -f ~/.ssh/swells_backup ]] && ! pgrep -f "backup_a100_checkpoints.sh" >/dev/null; then
  SOURCE=./$OUT DEST=~/a100-checkpoints/$RUN \
    nohup bash backup_a100_checkpoints.sh > "$OUT/backup.log" 2>&1 < /dev/null &
fi
if ! pgrep -f "capture_epoch1_ckpt.sh" >/dev/null; then
  RUN=./$OUT EPOCH_STEPS=1408 \
    nohup bash capture_epoch1_ckpt.sh > "$OUT/capture_epoch1.log" 2>&1 < /dev/null &
fi

for attempt in $(seq 1 20); do
  if bash run_curriculum_grpo.sh >> "$OUT/train.log" 2>&1; then
    echo "GRPO_8B_COT_DONE $(date -u)" >> "$OUT/train.log"
    break
  fi
  echo "runner exited with error (attempt $attempt), restarting in 60s" >> "$OUT/train.log"
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  pkill -f 'trl vllm-serve' 2>/dev/null || true
  sleep 60
done
