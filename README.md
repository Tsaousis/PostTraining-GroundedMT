# PostTraining-GroundedMT

Supervised fine-tuning versus reinforcement learning (GRPO, chrF reward) for translating unseen low-resource languages from grammar books and dictionaries.

**What does post-training teach? Supervised fine-tuning versus reinforcement learning for
resource-grounded translation of unseen languages.**

Code to reproduce every experiment, table and figure of the paper: SFT versus GRPO
(reinforcement learning with a verifiable chrF reward) for translating unseen languages from
grammar-book excerpts and dictionary entries, on the SWELLS artificial languages and on the
18 real low-resource languages of Back-to-School. No data is redistributed; the two datasets
are obtained from their authors (see *Data*).

Paper link: *to be added.*

## Contents

| Path | What it does |
|---|---|
| `swells/` | SWELLS: data preparation, curriculum schedule, SFT, GRPO (chrF, structured-grader and mixed-reward variants), evaluation, scoring, LoRA merge |
| `real_language/` | Back-to-School: prompt building (book+dict / dict / none / +retrieved examples), MIK training sets (direct, CoT, SWELLS-format), SFT, GRPO (4B direct, 4B-Instruct forced-CoT, 8B thinking), Llama-3.2-3B SFT, Gemini baseline, evaluation, aggregation, bootstrap |
| `counterfactual/` | Grammar-rule counterfactuals: adjective order, article position, plural morphology, constituent order (prompt sets included: artificial languages only) |
| `analysis/` | Reward-signal pilots, error profiles, lexical/morpheme credit split, per-rule accuracy, bootstrap CIs, training-curve and reward figures |
| `results/` | Aggregate tables behind the paper (real-language per-language chrF, counterfactual scores, reward pilots, interpretability) |
| `figures/` | Paper figures (PDF) |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt           # torch, transformers 4.56, trl 0.21, peft, vllm 0.10.1, sacrebleu
pip install flash-attn --no-build-isolation   # optional
```
`swells/setup_vm.sh` is the exact bootstrap used on the A100 VMs (CUDA 12, Ubuntu 22.04).
All commands below run from the repository root with the venv active, one or two A100-80GB.

## Data

**SWELLS** (Marmonier et al.): clone the official repository and dataset into `data/`:
```
data/SWELLS_Dataset/        # official release (Train/, Test/Test_CoT/, ...)
data/swells-official/       # official generator repo (used to rebuild the curriculum)
```
**Back-to-School** (Hus et al., 2024): the archive is available from its authors on request
and is **not** included here. Unpack it so that `real_language/data/back-to-school/lang/<language>/`
exists (or set `B2S_ROOT`). Only aggregate scores derived from it are in `results/real_language/`.

**MTOB / Kalamang** (Tanzer et al.) is used only by `real_language/build_mtob_prompts.py`.

## 1. SWELLS (Tables 1–3, Appendix A–B, E)

```bash
# Curriculum: 7 partition courses (276/1292/754/370/286/326/376 items) + mixed final stage (1,408)
python swells/build_swells_schedule.py --dataset data/SWELLS_Dataset --repo data/swells-official --output data/swells_schedule
python swells/prepare_nocot.py --in_file data/swells_schedule/final/train.jsonl --out_file data/swells_schedule/nocot/final/train.jsonl   # direct-answer prompts for the 4B GRPO arms (run per course)

# 4B SFT (CoT targets, 3 epochs, LoRA r=32, lr 1e-4)
MODEL=Qwen/Qwen3-4B-Base DATA=data/swells_schedule OUT=runs/sft_curriculum bash swells/run_curriculum_sft.sh
# 4B GRPO from base (direct answer, chrF reward, 8 rollouts, lr 1e-5, beta 0, eps_high 0.28, 2 epochs on the final stage)
MODEL=Qwen/Qwen3-4B-Base DATA=data/swells_schedule/nocot OUT=runs/grpo_curriculum_base DISABLE_THINKING=1 bash swells/run_curriculum_grpo.sh
# 4B GRPO from SFT
MODEL=runs/sft_curriculum/final/merged DATA=data/swells_schedule/nocot OUT=runs/grpo_curriculum_sft DISABLE_THINKING=1 bash swells/run_curriculum_grpo.sh
# 8B-Instruct SFT and GRPO in thinking mode (CoT curriculum, completion cap 4,096, 2 x A100)
bash swells/launch_qwen8b_instruct_sft_after_base.sh
bash swells/launch_qwen8b_instruct_cot_grpo.sh

# Evaluation: greedy, stop at </translation>, first tagged span; W+G test files, 100 items per partition and direction
python swells/evaluate.py --model runs/grpo_curriculum_base/final/merged --test_files data/SWELLS_Dataset/Test/Test_CoT/*grammar_excerpts.jsonl --output_dir eval/grpo4b --max_tokens 4096
bash swells/eval_8b_thinking.sh                   # 8B with thinking enabled, 16k cap
python swells/rescore.py eval/grpo4b              # exact accuracy and chrF per partition (held-out Latin partitions 8-9 reported separately)
python swells/eval_gemini.py --model gemini-3.1-pro-preview --test_files ... --output_dir eval/gemini   # GEMINI_API_KEY
```
Resource controls (Appendix A: task-only, W, W+B, W+G) use the corresponding official test
files with the same `evaluate.py` call. `swells/eval_curriculum_sft_epochs.sh` evaluates the
per-epoch checkpoints.

## 2. Real languages (Tables 4–5, Appendix C)

```bash
python real_language/build_prompts.py                       # 18 languages x {book_dict, dict, none} x 2 directions (B2S_CONTEXTS to change)
python real_language/build_train.py mizo ilokano kimbundu   # 5,982 MIK training prompts, book_dict context, direct targets
python real_language/build_train_swellsfmt.py <train.jsonl> <out.jsonl>   # SWELLS-format CoT prompts (1,998-prompt subset used for the 8B and CoT arms)
python real_language/build_cot_train.py <train.jsonl> <out.jsonl>         # forced-CoT variant for 4B-Instruct

bash real_language/run_sft_b2s.sh                           # 4B SFT, 3 epochs
python swells/grpo_train.py --model Qwen/Qwen3-4B-Base --train_file real_language/data/train/b2s_train_mizo_ilokano_kimbundu.jsonl \
   --output_dir runs/grpo_base --lora --epochs 1 --num_generations 8 --batch_size 1 --grad_accum 8 \
   --max_prompt_length 14336 --max_completion_length 192 --lr 1e-5 --beta 0 --epsilon_high 0.28 --disable_thinking   # 4B GRPO (from base or from the SFT merge)
bash real_language/run_grpo_4b_instruct_forced_cot.sh       # 4B-Instruct, CoT required (Appendix C)
bash real_language/run_grpo_8b_real_cot.sh                  # 8B thinking-mode GRPO on 1,998 SWELLS-format prompts (vLLM server + trainer)
bash real_language/run_llama_sft.sh                         # Llama-3.2-3B-Instruct SFT control
python swells/merge_lora.py --base <base> --adapter runs/<run>/final --out runs/<run>/merged

python real_language/eval_b2s.py --model runs/grpo_base/merged --tag grpo4b-base --completion          # direct-answer models
python real_language/eval_b2s.py --model <8B merged> --tag grpo8b --prompts real_language/data/prompts_swellsfmt --cot --enable_thinking --max_tokens 4096   # thinking models, also for the SWELLS->real transfer (Table 5)
python real_language/aggregate.py <tag>                     # per-language / per-context chrF, dictionary and grammar-book contributions
python real_language/hus_8b_boot.py; python analysis/hus_bootstrap.py   # paired bootstrap over languages (Table 5 stars, Appendix CIs)
python real_language/run_gemini.py                          # Gemini 3.1 Pro baseline
bash real_language/run_ex_evals.sh                          # +5 retrieved parallel examples ablation (build_prompts_retr.py / build_train_ex.py, score_retr_full.py)
```

## 3. Counterfactual grammar tests (Section 6.1, Appendix D)

Prompt sets are included (they contain only SWELLS artificial-language text):
`counterfactual/cf_{cot,nocot}_eng_art.jsonl` (adjective order, 600 items),
`article/cf_article_cot_eng_art.jsonl` (300), `plural/cf_plural_cot_eng_art.jsonl` (233),
`wordorder/cf_wo_cot_eng_art.jsonl` (140 after exclusions). To rebuild: `build_cf*.py <test.jsonl> <out.jsonl>`.
```bash
python swells/evaluate.py --model <merged model> --test_files counterfactual/cf_cot_eng_art.jsonl --output_dir cf/<arm> --max_tokens 4096
python counterfactual/score_cf.py cf/<arm>; python counterfactual/score_orig.py cf/<arm>; python counterfactual/engorder.py; python counterfactual/bags_split.py
# article/, plural/, wordorder/: same pattern with their score_*.py (fuzzy token match >= 0.6)
```

## 4. Reward pilots and analyses (Section 6.2, Appendix E–F)

```bash
REWARD=v2 COURSES=4 bash swells/launch_grader_v2_pilot.sh     # structured grader (lex/rule/order/fmt), 8B or 4B, partition 4
REWARD=chrf_wordexact bash swells/launch_4b_mixed_reward.sh   # chrF + word-exact term, partition 7
python analysis/reward_signal_check.py --model <model> --train_file <course jsonl> --out <json>   # 8 rollouts per prompt: how often a group has an exact rollout, chrF gap correct vs near-miss
python analysis/rescore_signal_graders.py; python analysis/summarize_cot_signal.py; python analysis/pilot_variance.py
python analysis/paper_swells_errors.py        # error types over W+G art->eng failures (Table 7)
python analysis/paper_hus_errors.py           # real-language error profile, tag rate, loop / cap statistics
python analysis/paper_interp.py               # lexical vs morpheme credit split, per-rule accuracy, trace-grounding regression
python analysis/make_training_curves.py; python analysis/paper_reward_figs.py   # Figures 2-3
```
Analysis scripts read the evaluation outputs produced above; the input paths are set at the
top of each script.

## Results

- `results/real_language/`: per-language chrF of every arm (`hus_chrf_*.tsv`), 8B tables,
  bootstrap CIs, prompt-format 2x2, GRPO data-size curve.
- `results/counterfactual/`: adjective-order, article, word-order scores and notes.
- `results/reward_signal/`: per-group reward statistics of the pilots.
- `results/interp/`: credit split, feature accuracy/logit, trace grounding.

Raw predictions, per-rollout reward logs and LoRA adapters are released separately
(Hugging Face, link to be added).

## Hyperparameters

LoRA r=32, alpha=64 on all attention and MLP projections; SFT lr 1e-4, 3 epochs; GRPO lr 1e-5,
8 rollouts, temperature 1, no KL (beta 0), epsilon_high 0.28, completion cap 2,048 (4,096 for
the 8B SWELLS run); greedy decoding at test time with a 4k (4B) or 16k (8B) cap. One run per
configuration. Compute: ~15 GPU-hours for a 4B direct-answer GRPO pass over 5,982 prompts,
~20 GPU-hours for the 8B thinking-mode pass over 1,998 prompts.

## Citation

To be added.

## License

MIT (code). SWELLS and Back-to-School data remain under their own licenses.
