#!/usr/bin/env bash
# On the base A100: wait for GRPO-from-base to finish and be backed up, then run
# Qwen3-8B (official instruct) curriculum SFT on the frozen SWELLS schedule.
# 1 GPU -> grad_accum 16 keeps the effective batch of 16 used for the 4x L4 4B run.
# SFT keeps the dataset's explicit CoT; DISABLE_THINKING only selects Qwen3's
# non-thinking template marker so prompt/label masking matches evaluation.
set -u
cd .
source venv/bin/activate

RUN=${RUN:-sft_curriculum_qwen8b_instruct}
BASE_OUT=runs/grpo_curriculum_base

until [[ -s $BASE_OUT/final/merged/config.json ]] && ! pgrep -f 'grpo_train.py' >/dev/null; do
  sleep 120
done
# give the rolling backup a few cycles to ship the final merged model
sleep 900
pkill -9 -f 'VLLM::' 2>/dev/null || true

mkdir -p runs/$RUN
SOURCE=./runs/$RUN DEST=~/a100-checkpoints/$RUN \
  nohup bash backup_a100_checkpoints.sh > runs/$RUN/backup.log 2>&1 < /dev/null &

for attempt in $(seq 1 10); do
  if MODEL=Qwen/Qwen3-8B OUT=runs/$RUN GPUS=1 GRAD_ACCUM=16 DISABLE_THINKING=1 \
      bash run_curriculum_sft.sh >> runs/$RUN/train.log 2>&1; then
    echo "SFT_8B_DONE $(date -u)" >> runs/$RUN/train.log
    break
  fi
  echo "sft runner exited with error (attempt $attempt), restarting in 60s" >> runs/$RUN/train.log
  sleep 60
done
