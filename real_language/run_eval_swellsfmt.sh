#!/bin/bash
# GPU1: untrained Qwen3-4B-Base on Hus test items re-rendered in SWELLS CoT format
source venv/bin/activate; cd real_language
TAG=qwen3-4b-base; MODEL=Qwen/Qwen3-4B-Base; EXP=6668
CUDA_VISIBLE_DEVICES=1 setsid python real_language/eval_b2s.py --model $MODEL --tag $TAG --prompts real_language/data/prompts_swellsfmt --out real_language/data/outputs_swellsfmt --cot --max_tokens 4096 --max_model_len 8192 > real_language/data/logs_evalsf_$TAG.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  if grep -q ALL_DONE real_language/data/logs_evalsf_$TAG.log; then sleep 60; kill -- -$PID 2>/dev/null; break; fi
  sleep 60
done
N=$(cat real_language/data/outputs_swellsfmt/$TAG/*.jsonl 2>/dev/null | wc -l)
[ "$N" = "$EXP" ] && echo OK_EVALSF_$TAG >> real_language/data/status.txt || echo FAIL_EVALSF_${TAG}_rows$N >> real_language/data/status.txt
