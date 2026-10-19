#!/usr/bin/env bash
# Qwen3-8B instruct, thinking ON, SWELLS W+G CoT test, 32k ctx / 16k gen cap.
set -u
cd swells; source venv/bin/activate; export VLLM_WORKER_MULTIPROC_METHOD=spawn
while pgrep -f "eval_grpo_nocot|evaluate.py|eval_translation_tags" >/dev/null; do sleep 60; done
T=~/swells_min/SWELLS_Dataset/Test/Test_CoT
[[ -s eval/qwen3_8b_instruct_think_wg_16k/summary.json ]] || python evaluate.py --model Qwen/Qwen3-8B --enable_thinking \
  --test_files $T/CoT_full_art_eng_test_grammar_excerpts.jsonl $T/CoT_full_eng_art_test_grammar_excerpts.jsonl \
  --output_dir eval/qwen3_8b_instruct_think_wg_16k --tp 4 --max_model_len 32768 --max_tokens 16384 --max_num_seqs 16 --gpu_util 0.92
echo THINK_EVAL_DONE
