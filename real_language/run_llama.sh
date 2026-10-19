#!/bin/bash
# GPU1 chain: Llama-3.2-3B-Instruct on the same MIK data as the Qwen arms.
# untrained Hus eval -> SFT (5,982 prompts, 3 ep) -> merge -> chrF-GRPO from Llama (same 5,982, 1 ep) -> merge -> Hus evals -> stop VM when GPU0 chain also done
cd swells && source venv/bin/activate
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=unsloth/Llama-3.2-3B-Instruct
TRAIN=real_language/data/train/b2s_train_mizo_ilokano_kimbundu.jsonl
G=1
ev() { CUDA_VISIBLE_DEVICES=$G python real_language/eval_b2s.py --model $1 --tag $2 --suffix > real_language/data/logs_eval_$2.log 2>&1 && echo OK_EVAL_$2 >> real_language/data/status.txt || echo FAIL_EVAL_$2 >> real_language/data/status.txt; }
ev $M llama3b-base
SFT=real_language/data/runs/llama_sft_mik; mkdir -p $SFT
CUDA_VISIBLE_DEVICES=$G python sft_train.py --model $M --train_file $TRAIN --output_dir $SFT --lora --epochs 3 --per_device_bs 1 --grad_accum 16 --lr 1e-4 --max_len 10240 --fail_on_skip > real_language/data/logs_llama_sft.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=$G python merge_lora.py --base $M --adapter $SFT/final --out $SFT/merged >> real_language/data/logs_llama_sft.log 2>&1 && echo OK_llama_sft >> real_language/data/status.txt || { echo FAIL_llama_sft >> real_language/data/status.txt; }
GR=real_language/data/runs/llama_grpo_base_full; mkdir -p $GR
CUDA_VISIBLE_DEVICES=$G SWELLS_REWARD_LOG=$GR/rewards.jsonl python grpo_train.py --model $M --train_file $TRAIN --output_dir $GR --lora --epochs 1 --num_generations 8 --batch_size 1 --grad_accum 8 --max_prompt_length 14336 --max_completion_length 192 --lr 1e-5 --beta 0.0 --epsilon_high 0.28 --vllm_mode colocate --vllm_gpu_util 0.35 --vllm_server_port 8013 > $GR.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=$G python merge_lora.py --base $M --adapter $GR/final --out $GR/merged >> $GR.log 2>&1 && echo OK_llama_grpo_base_full >> real_language/data/status.txt || echo FAIL_llama_grpo_base_full >> real_language/data/status.txt
[ -s $SFT/merged/config.json ] && ev $SFT/merged llama3b-sft
[ -s $GR/merged/config.json ] && ev $GR/merged llama3b-grpo-base-full
echo LLAMA_DONE >> real_language/data/status.txt
until grep -q FULL_DONE real_language/data/status.txt; do sleep 120; done
sudo poweroff
