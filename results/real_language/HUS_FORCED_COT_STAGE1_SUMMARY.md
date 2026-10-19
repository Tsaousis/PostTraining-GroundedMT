# Forced-CoT chrF-GRPO, stage 1 (1,998 MIK prompts) — Hus eval, 18 langs × 2 dirs × 3 contexts (10,002 rows)

Model: Qwen3-4B instruct (thinking on), LoRA GRPO, clean forced-reasoning prompt (reward 0 unless ≥200 reasoning chars precede `<translation>`), max completion 2,048 tokens at train and eval. Evaluated 2026-10-09 on kostas; raw outputs in ~/b2s-eval/outputs/grpo4b-forced-s1 (36 MB). Per-language table: results/HUS_B2S_QWEN4B_GRPO_FORCED_S1.tsv.

Mean chrF (tag rate in parentheses), compared with the no-CoT GRPO-from-base at the same 1,998 prompts (grpo4b-base sub2k) and the untrained base:

| arm | seen en→X | seen X→en | unseen en→X | unseen X→en |
|---|---|---|---|---|
| Qwen3-4B-Base, book+dict | 12.7 | 29.0 | 13.4 | 25.1 |
| no-CoT GRPO sub2k, book+dict | 30.6 | 31.2 | 22.2 | 30.4 |
| forced-CoT s1, book+dict | 19.7 (63%) | 35.9 (99%) | 16.7 (65%) | 33.3 (99%) |
| forced-CoT s1, dict | 20.8 (69%) | 36.1 (99%) | 17.3 (68%) | 33.4 (99%) |
| forced-CoT s1, none | 17.3 (67%) | 36.1 (100%) | 16.5 (72%) | 32.8 (100%) |

Findings
- X→en: forced-CoT beats no-CoT GRPO by ~+3 chrF on both seen and unseen languages (35.9/33.3 vs 31.2/30.4).
- en→X: ~1/3 of outputs never emit a translation tag — the reasoning degenerates into a repetition loop (e.g. "kaka kaka …") until the 2,048-token cap and scores 0. Restricted to tagged outputs, unseen en→X is 23.2 (≈ no-CoT 22.2); seen 28.2.
- Context ablation is flat for forced-CoT (book+dict ≈ dict ≈ none, within ±1), unlike no-CoT GRPO (dict − none = +2.9 unseen en→X). The reasoning traces cite dictionary entries, but the resources do not move the score at this stage.
- Stage 2 (remaining 3,984 prompts) is running; re-evaluate the final model before drawing conclusions. The en→X loop failure needs a fix (repetition penalty at eval, longer cap, or a loop penalty in the reward) before this arm can be compared fairly.
