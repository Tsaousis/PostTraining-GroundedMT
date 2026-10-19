#!/usr/bin/env bash
# Ablation: 4B-Base final stage, reward 0.5*chrF + 0.5*word-exact (REWARD=chrf_wordexact),
# resumed from the chrF-GRPO course_7 merged model: identical data schedule, final stage only,
# FINAL_EPOCHS epochs, direct/no-CoT. Earlier stages are symlinked so run_curriculum_grpo.sh skips them.
set -euo pipefail
cd swells && source venv/bin/activate
export REWARD=${REWARD:-chrf_wordexact}
export OUT=runs/grpo_curriculum_base_${REWARD}
export DATA=${DATA:-${SWELLS_SCHEDULE:-data/swells_schedule}/nocot}
export MODEL=Qwen/Qwen3-4B-Base
export MODE=colocate
export GPU_UTIL=${GPU_UTIL:-0.35}
export TRAIN_VISIBLE=${TRAIN_VISIBLE:-1}
export TRAIN_PROCESSES=${TRAIN_PROCESSES:-1}
export FINAL_EPOCHS=${FINAL_EPOCHS:-2}
export SWELLS_REWARD_LOG=$PWD/$OUT/reward_log.jsonl
mkdir -p "$OUT"

[[ -s runs/grpo_curriculum_base/course_7/merged/config.json ]] || { echo "missing course_7/merged"; exit 1; }
for c in 1 2 3 4 5 6 7; do
  [[ -e "$OUT/course_$c" ]] || ln -s "$PWD/runs/grpo_curriculum_base/course_$c" "$OUT/course_$c"
done
# only course_7/merged must be real; courses 1-6 are skipped iff their merged/config.json exists
for c in 1 2 3 4 5 6; do
  [[ -s "$OUT/course_$c/merged/config.json" ]] || { mkdir -p "runs/grpo_curriculum_base/course_$c/merged"; echo '{"placeholder": true}' > "runs/grpo_curriculum_base/course_$c/merged/config.json"; }
done
exec bash run_curriculum_grpo.sh
