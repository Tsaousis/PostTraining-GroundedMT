#!/bin/bash
# GPU1, after Llama SFT chain: evaluate on Hus-style "+5 LCS-retrieved examples" prompts (book_dict_ex / dict_ex), outputs under real_language/data/outputs_ex
cd swells && source venv/bin/activate
P=real_language/data/prompts_ex; O=real_language/data/outputs_ex; mkdir -p $O
until grep -q LLAMA_SFT_DONE real_language/data/status.txt; do sleep 60; done
ev() { CUDA_VISIBLE_DEVICES=1 python real_language/eval_b2s.py --model $1 --tag $2 --prompts $P --out $O $3 > real_language/data/logs_evalex_$2.log 2>&1 && echo OK_EVALEX_$2 >> real_language/data/status.txt || echo FAIL_EVALEX_$2 >> real_language/data/status.txt; }
[ -s real_language/data/runs/llama_sft_mik/merged/config.json ] && ev real_language/data/runs/llama_sft_mik/merged llama3b-sft --suffix
ev real_language/data/runs/sft_mizo_ilo_kim/merged sft4b --suffix
ev unsloth/Llama-3.2-3B-Instruct llama3b-base --suffix
ev Qwen/Qwen3-4B-Base qwen3-4b-base --completion
ev real_language/data/runs/grpo_base/merged grpo4b-base --suffix
echo EX_EVALS_DONE >> real_language/data/status.txt
