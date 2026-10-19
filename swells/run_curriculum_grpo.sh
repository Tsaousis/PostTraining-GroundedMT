#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-4B-Base}
DATA=${DATA:-${SWELLS_SCHEDULE:-data/swells_curriculum/sft}/nocot}
OUT=${OUT:-runs/grpo_curriculum_base}
MODE=${MODE:-colocate}
GPU_UTIL=${GPU_UTIL:-0.35}
PORT=${PORT:-8000}
FINAL_EPOCHS=${FINAL_EPOCHS:-2}
DISABLE_THINKING=${DISABLE_THINKING:-0}
TRAIN_VISIBLE=${TRAIN_VISIBLE:-0,1}
TRAIN_PROCESSES=${TRAIN_PROCESSES:-2}
VLLM_VISIBLE=${VLLM_VISIBLE:-2,3}
VLLM_TP=${VLLM_TP:-1}
VLLM_DP=${VLLM_DP:-2}
REWARD=${REWARD:-chrf}            # chrf | v2 (grader_v2, needs META_DIR)
META_DIR=${META_DIR:-${SWELLS_SCHEDULE:-data/swells_curriculum/sft}/meta}
COURSES=${COURSES:-1 2 3 4 5 6 7}  # e.g. COURSES=4 FINAL_EPOCHS=0 for a single-course pilot
MAX_COMPLETION=${MAX_COMPLETION:-192}  # raise (e.g. 1024) with DATA=CoT schedule to keep reasoning in rollouts

mkdir -p "$OUT"
current_model=$MODEL
thinking_args=()
if [[ "$DISABLE_THINKING" == 1 ]]; then
  thinking_args=(--disable_thinking)
fi

run_stage() {
  local name=$1
  local file=$2
  local epochs=$3
  local stage="$OUT/$name"
  mkdir -p "$stage"

  if [[ ! -s "$stage/merged/config.json" ]]; then
    local resume=()
    local checkpoint
    checkpoint=$(find "$stage" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
      | sort -V | tail -1)
    if [[ -n "$checkpoint" ]]; then
      resume=(--resume_from_checkpoint "$checkpoint")
    fi

    args=(
      --model "$current_model" \
      --train_file "$file" \
      --output_dir "$stage" \
      --lora \
      --epochs "$epochs" \
      --num_generations 8 \
      --batch_size 1 \
      --grad_accum 8 \
      --max_prompt_length 14336 \
      --max_completion_length "$MAX_COMPLETION" \
      --lr 1e-5 \
      --beta 0.0 \
      --epsilon_high 0.28 \
      --vllm_mode "$MODE" \
      --vllm_gpu_util "$GPU_UTIL" \
      --vllm_server_port "$PORT" \
      "${thinking_args[@]}" \
      "${resume[@]}"
    )
    if [[ "$REWARD" == v2 ]]; then
      args+=(--reward v2 --meta_file "$META_DIR/$(basename "$file")")
    fi

    if [[ "$MODE" == server ]]; then
      CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE" trl vllm-serve \
        --model "$current_model" \
        --port "$PORT" \
        --tensor-parallel-size "$VLLM_TP" \
        --data-parallel-size "$VLLM_DP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len "$((14336 + MAX_COMPLETION))" \
        > "$stage/vllm.log" 2>&1 &
      server_pid=$!
      for _ in $(seq 1 120); do
        curl -sf "http://localhost:$PORT/health/" >/dev/null && break
        sleep 10
      done
      CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE" accelerate launch \
        --num_processes "$TRAIN_PROCESSES" \
        --mixed_precision bf16 \
        grpo_train.py "${args[@]}" --liger_loss
      kill "$server_pid" 2>/dev/null || true
      pkill -9 -f 'VLLM::' 2>/dev/null || true
      sleep 20
    else
      python grpo_train.py "${args[@]}"
    fi

    python merge_lora.py \
      --base "$current_model" \
      --adapter "$stage/final" \
      --out "$stage/merged"
  fi
  current_model="$stage/merged"
}

for course in $COURSES; do
  run_stage "course_$course" "$DATA/course_$course.jsonl" 1
done
if [[ "$FINAL_EPOCHS" != 0 ]]; then
  run_stage final "$DATA/final_1408.jsonl" "$FINAL_EPOCHS"
  echo "$OUT/final/merged"
else
  echo "$current_model"
fi
