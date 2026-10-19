# All results so far (2026-09-30, 19:10 UTC)

Full write-up: `paper/RESULTS.md`; findings + plan: `paper/NOTES_AND_PLAN.md`. Pending: GRPO-from-base 2-epoch final and GRPO-from-SFT 1-epoch final (both ~07:00 Athens, 1 Oct).

## 1. SWELLS W+G test, 1,400 items/direction, 16k context, 4,096-token cap, official SWELLS accuracy

| Model | art→eng acc | art→eng chrF | art→eng EM | eng→art acc | eng→art chrF | eng→art EM |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla Qwen3-4B-Base | 6.4 | 45.82 | 1.2% | 1.5 | 46.54 | 1.3% |
| Vanilla Qwen3-8B-Base (scale ref.) | 19.2 | 60.19 | 8.1% | 6.3 | 61.92 | 5.7% |
| Qwen3-8B instruct, thinking off (untrained ref.) | 27.6 | 70.48 | 9.3% | 11.9 | 70.72 | 10.4% |
| Curriculum SFT 4B (CoT) | **41.2** | **73.29** | 40.6% | **23.6** | **71.83** | 22.6% |
| GRPO-from-base 4B, 1-epoch final (no-CoT) | 13.3 | 61.79 | 3.0% | 2.2 | 52.69 | 2.1% |
| GRPO-from-base 4B, 2-epoch final | pending | | | | | |
| GRPO-from-SFT 4B, 1-epoch final (no-CoT) | 13.2 | 60.93 | 0.0% | 2.2 | 56.00 | 0.0% |
| Gemini 3.1 Pro, thinking on (frontier ref., API, 1 Oct) | 96.7 | 96.50 | 41.6% | 96.4 | 96.72 | 77.9% |
| o4-mini untrained, reasoning medium (OpenAI API, 5 Oct) | 75.0 | 91.13 | 32.6% | 50.6 | 90.31 | 43.7% |
| GPT-4o-mini baseline (SWELLS paper, CoT) | 32.4 | – | – | 9.5 | – | – |
| GPT-4o-mini baseline (SWELLS paper, no-CoT) | 31.2 | – | – | 5.9 | – | – |
| GPT-4o-mini fine-tuned (SWELLS paper) | 61.4 | – | – | 62.9 | – | – |

Caveat (all GRPO rows): GRPO is trained with the SWELLS *no-CoT* prompt but these rows use the *CoT* test prompt (same as every other row). The GRPO models still answer directly (mean completion ~60 chars). A no-CoT-prompt W+G eval of both GRPO 1-epoch checkpoints is queued on the L4 to rule out a prompt-mismatch effect.

Missing `</translation>` (of 1,400, art→eng / eng→art): base 4B 11 / 49; base 8B 307 / 47; 8B instruct 42 / 8; curriculum SFT 1 / 1; GRPO-from-base 8 / 58.

## 1b. Gemini 3.1 Pro (thinking on, temp 0, Vertex API) — per-partition accuracy on the W+G test

Same 2,800 items/scoring as table 1; 0 API errors; ~11.6M prompt + ~0.7M answer + ~3.3–4.5M thought tokens per direction (~$90 total). Raw outputs: `results/gemini31pro_wg_4k/`.

| Partition | 1 | 2f | 2m | 3 | 4 | 5f | 5m | 6 | 7f | 7m | 8a | 8g | 9a | 9g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| art→eng | 100 | 100 | 100 | 100 | 100 | 99 | 99 | 92 | 84 | 82 | 100 | 98 | 100 | 100 |
| eng→art | 100 | 95 | 100 | 96 | 100 | 95 | 97 | 98 | 92 | 98 | 97 | 95 | 92 | 94 |

Reading: a frontier reasoning model solves SWELLS W+G almost completely (≈96–97% both directions), *including the Latin-derived held-out partitions 8–9 (92–100%)* that no fine-tuned model in this study sees. The weakest cells are partitions 6–7 art→eng (82–92%). This bounds the task from above: the grammar excerpts + dictionary contain enough information for exact rule application; the 4B/8B gap (SFT 41/24, 8B instruct 28/12) is model capability/reasoning, not benchmark noise or missing information.

## 1c. o4-mini untrained (OpenAI Responses API, reasoning effort medium) — per-partition accuracy on the W+G test

Same 2,800 items/scoring as table 1; 0 API errors, 0 truncated; ~11.4M prompt + ~0.3M answer + ~2.7–3.0M reasoning tokens per direction (~$20 total). Raw outputs: `results/o4mini_wg_4k/`, script `code/eval_openai.py`.

| Partition | 1 | 2f | 2m | 3 | 4 | 5f | 5m | 6 | 7f | 7m | 8a | 8g | 9a | 9g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| art→eng | 99 | 100 | 96 | 85 | 86 | 84 | 87 | 67 | 55 | 43 | 67 | 54 | 72 | 55 |
| eng→art | 100 | 83 | 94 | 68 | 52 | 39 | 40 | 7 | 5 | 10 | 83 | 41 | 63 | 23 |

Reading: a mid-size reasoning model sits between our 8B SFT (50.3 / 33.9) and Gemini (96.7 / 96.4): art→eng 75.0, eng→art 50.6. Its profile is the one RL needs — Latin held-out is strong (23–83) but partitions 6–7 are near zero eng→art (5–10%), i.e. a high plateau with a clear unsolved region. Candidate base for an API-RFT "stronger base" experiment (not part of the Qwen-only main study).

## 2. SWELLS W+G per partition, official accuracy

art→eng:

| partition | base 4B | base 8B | 8B instruct | curriculum SFT 4B | GRPO-from-base (1 ep.) | GRPO-from-SFT (1 ep.) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 37.0 | 56.0 | 100.0 | 77.0 | 60.0 | 64.0 |
| 2m | 16.0 | 64.0 | 79.0 | 91.0 | 31.0 | 35.0 |
| 2f | 32.0 | 63.0 | 66.0 | 82.0 | 48.0 | 47.0 |
| 3 | 3.0 | 13.0 | 33.0 | 33.0 | 8.0 | 3.0 |
| 4 | 0.0 | 7.0 | 19.0 | 28.0 | 0.0 | 1.0 |
| 5m | 0.0 | 13.0 | 17.0 | 20.0 | 0.0 | 0.0 |
| 5f | 0.0 | 7.0 | 10.0 | 22.0 | 0.0 | 0.0 |
| 6 | 0.0 | 3.0 | 4.0 | 72.0 | 2.0 | 1.0 |
| 7m | 1.0 | 4.0 | 2.0 | 68.0 | 5.0 | 2.0 |
| 7f | 0.0 | 2.0 | 3.0 | 68.0 | 1.0 | 2.0 |
| 8a (Latin, held-out) | 0.0 | 12.0 | 21.0 | 8.0 | 9.0 | 6.0 |
| 8g (Latin, held-out) | 1.0 | 9.0 | 10.0 | 3.0 | 13.0 | 11.0 |
| 9a (Latin, held-out) | 0.0 | 7.0 | 16.0 | 3.0 | 2.0 | 2.0 |
| 9g (Latin, held-out) | 0.0 | 9.0 | 6.0 | 2.0 | 7.0 | 11.0 |
| **average** | **6.4** | **19.2** | **27.6** | **41.2** | **13.3** | **13.2** |

eng→art:

| partition | base 4B | base 8B | 8B instruct | curriculum SFT 4B | GRPO-from-base (1 ep.) | GRPO-from-SFT (1 ep.) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 | 42.0 | 66.0 | 45.0 | 13.0 | 13.0 |
| 2m | 3.5 | 8.0 | 24.5 | 40.0 | 6.5 | 5.5 |
| 2f | 2.0 | 4.5 | 8.5 | 44.0 | 1.0 | 2.0 |
| 3 | 9.0 | 27.0 | 37.0 | 42.0 | 10.0 | 7.0 |
| 4 | 0.0 | 2.0 | 11.0 | 31.0 | 0.0 | 0.0 |
| 5m | 0.0 | 0.5 | 1.0 | 24.5 | 0.0 | 0.0 |
| 5f | 0.0 | 0.0 | 1.0 | 21.0 | 0.0 | 0.0 |
| 6 | 0.0 | 0.0 | 0.0 | 27.0 | 0.0 | 0.0 |
| 7m | 0.0 | 0.0 | 0.0 | 23.5 | 0.0 | 0.0 |
| 7f | 0.0 | 0.0 | 0.0 | 31.0 | 0.0 | 0.0 |
| 8a (Latin, held-out) | 0.0 | 4.0 | 17.0 | 1.0 | 0.0 | 1.0 |
| 8g (Latin, held-out) | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 2.0 |
| 9a (Latin, held-out) | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| 9g (Latin, held-out) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **average** | **1.5** | **6.3** | **11.9** | **23.6** | **2.2** | **2.2** |

## 3. GRPO training dynamics (chrF reward, start → end of stage)

GRPO-from-base:

| Stage | chrF reward | EM diag. | mean completion len | frac. zero-variance groups |
|---|---|---|---:|---:|
| Course 1 | 0.56 → 0.89 | 0.07 → 0.27 | 14 | 0.73 |
| Course 2 | 0.73 → 0.82 | 0.05 → 0.14 | 22 | 0.53 |
| Course 3 | 0.62 → 0.78 | 0.06 → 0.30 | 18 | 0.58 |
| Course 4 | 0.66 → 0.72 | 0.00 → 0.00 | 27 | 0.34 |
| Course 5 | 0.72 → 0.73 | 0.00 → 0.00 | 35 | 0.16 |
| Course 6 | 0.57 → 0.59 | 0.00 → 0.00 | 29 | 0.15 |
| Course 7 | 0.60 → 0.63 | 0.01 → 0.02 | 37 | 0.08 |
| Final (2 ep.) | running (epoch 1 done) | | | |

GRPO-from-SFT:

| Stage | chrF reward | EM diag. | mean completion len | frac. zero-variance groups |
|---|---|---|---:|---:|
| Course 1 | 0.91 → 0.94 | 0.64 → 0.72 | 132 | 0.51 |
| Course 2 | 0.87 → 0.92 | 0.37 → 0.44 | 57 | 0.52 |
| Course 3 | 0.85 → 0.86 | 0.45 → 0.55 | 37 | 0.73 |
| Course 4 | 0.89 → 0.90 | 0.33 → 0.30 | 46 | 0.43 |
| Course 5 | 0.88 → 0.88 | 0.16 → 0.19 | 60 | 0.42 |
| Course 6–7, Final | running (final stage ~30%) | | | |

## 4. Kalamang transfer (Hu et al. exact test parquet + evaluator; chrF, eng→kgv / kgv→eng)

| Model | Hu prompt, with resources | Hu prompt, task-only | SWELLS-format, with resources | SWELLS-format, task-only |
|---|---|---|---|---|
| Hu et al.: Qwen3-4B-Base | 25.6 / 27.5 | ~15 | – | – |
| Hu et al.: + Romansh SFT | 28.6 / 35.2 | | – | – |
| Hu et al.: + Romansh RL | **34.6 / 38.4** | | – | – |
| Ours: Qwen3-4B-Base | 27.1 / 28.2 | 15.6 / 13.6 | 26.7 / 35.0 (CoT), 30.0 / 33.2 (no-CoT) | 13.9 / 14.6, 14.4 / 14.5 |
| Ours: curriculum SFT | **7.5 / 6.8** (collapse) | 8.0 / 5.4 | 22.3 / 24.0 (CoT) | 11.5 / 10.9 |
| Ours: GRPO-from-base, after course 7 | 25.0 / 30.6 | 15.3 / 15.0 | 27.2 / 32.9 (no-CoT) | 15.3 / 15.0 |
| Ours: GRPO-from-base, 1-epoch final | 25.4 / 28.7 | 16.6 / 12.6 | 27.9 / 34.7 (no-CoT) | 14.5 / 14.8 |
| Ours: GRPO-from-SFT, 1-epoch final | 25.7 / 28.1 | 16.4 / 14.3 | 29.4 / 34.9 (no-CoT) | 15.9 / 15.0 |
| Ours: GRPO-from-base, 2-epoch final (2026-10-02) | 27.1 / 29.6 (thinking on, Hu default); 25.2 / 29.7 (thinking off) | 16.1 / 13.9; 15.6 / 14.5 (thinking off) | 26.8 / 31.7 (CoT, thinking on) | 15.5 / 16.5 |

Note (2-epoch final): the checkpoint that jumped to 42.4 / 6.1 on SWELLS stays at base level on Kalamang under every prompt condition — Hu prompt with native thinking on or off, and the SWELLS CoT prompt with thinking on. Raw: results/kalamang/grpo_base_ep2*__summary.json / __details.parquet (A100-40GB, vLLM TP1, 16k cap, greedy).

Note: chrF-GRPO on top of the collapsed SFT model (7.5 / 6.8) restores Kalamang transfer to base level (25.7 / 28.1), while its SWELLS accuracy falls from 41.2 / 23.6 to 13.2 / 2.2 — i.e. GRPO-from-SFT ends up at the same point as GRPO-from-base on both benchmarks.

## 4b. Grader-v2 reward pilot (course 4 only, 1 epoch, Qwen3-4B-Base; test = partition 4, 100 items/direction)

Training signal (per-rollout `results/pilot_v2/*_course4_rewards_v2.jsonl`, 2,960 rollouts each, means over 5 equal windows):

| Pilot | total | lex | rule | order | diag. chrF | exact | s/step |
|---|---|---|---|---|---|---|---|
| no-CoT rollouts (192-tok cap) | 0.16 → 0.49 | 0.53 → 0.97 | 0.15 → 0.35 | 0.22 → 0.39 | 0.33 → 0.60 | 0 / 2,960 | 38 |
| CoT rollouts (1,024-tok cap, CoT prompts) | 0.11 → 0.27 | 0.41 → 0.76 | 0.14 → 0.26 | 0.17 → 0.28 | 0.26 → 0.45 | 0 / 2,960 | 50 |

| Model (eval prompt) | art→eng acc / chrF | eng→art acc / chrF |
|---|---|---|
| Qwen3-4B-Base (no-CoT prompt) | 0.0 / 52.8 | 0.0 / 36.9 |
| Qwen3-4B-Base (CoT prompt) | 0.0 / 37.5 | 0.0 / 39.0 |
| chrF-GRPO-from-base, 1-epoch final, full curriculum (CoT prompt) | 0.0 / 54.7 | 0.0 / 49.4 |
| **v2-GRPO pilot, course 4 only, no-CoT (no-CoT prompt)** | **1.0 / 55.0** | **0.0 / 54.6** |
| **v2-GRPO pilot, course 4 only, CoT rollouts (CoT prompt)** | **0.0 / 39.2** | **0.0 / 43.6** |
| Curriculum SFT (CoT prompt), reference | 28.0 / 69.4 | 31.0 / 84.3 |
| **Pilot A**: v2 + CoT GRPO from curriculum-SFT 4B, course 4 (1 Oct) | 31.0 / 72.5 | 30.0 / 81.3 |
| Qwen3-8B instruct, thinking on, untrained (p4 of W+G eval) | 25.0 / 67.2 | 24.0 / 86.7 |
| Qwen3-8B instruct curriculum SFT (p4 of W+G eval) | 46.0 / 75.8 | 43.0 / 85.2 |
| **Pilot B**: v2 + CoT GRPO from Qwen3-8B instruct, course 4, 1,536-tok cap, 2×A100 server mode (2 Oct) | **15.0 / 66.5** | **14.0 / 80.4** |

Pilot B (2 Oct): grader total rose 0.08 → 0.67 (lex 0.41 → 0.95, rule 0.19 → 0.69, fmt 0.09 → 0.94) over 370 steps, yet held-out p4 exact fell below the *untrained* 8B instruct (15/14 vs 25/24; SFT 46/43). Predictions keep the lexicon and format but swap tense/aspect ("was watching" for "would watch", "will" for "would"): the policy optimised the lexical/format terms of the grader and lost rule application. Every output closes the tag (0 empty, 0 at cap). Decision: **no full-curriculum grader-v2 run**; the chrF-only CoT arm remains the sole 8B GRPO line. Raw: `results/raw/grpo_v2_pilot_course4_qwen8b_instruct_cot_p4_summary.json`.


Reading: the grader works as a dense reward (smooth climb, no affix spam, 0 flat groups) and the no-CoT pilot moves chrF on eng→art as much as the whole chrF curriculum did, but after one course the policy never produces the exact rule-ON form (rule score plateaus ≈0.35: it places a prefix on the wrong word). eng→art no-CoT outputs are systematically one word short (2.9 vs 3.9 tokens): the reward penalises extra tokens but under-penalises omissions — fix for any full run. Keeping CoT in the rollouts does **not** help from the base model: learning is slower (longer, noisier completions; 3% never close the tag), the reasoning text cites the right rule but mis-applies it (e.g. picks the wrong ending or pluralises the wrong token), and partition-4 accuracy stays 0 with chrF barely above base. Conclusion: one course of grammar-aware RL on Qwen3-4B-Base does not teach exact morphology with or without CoT; the SFT model's 28–31 on this partition comes from imitating worked solutions, which no reward signal we tried recovers from the base in 370 prompts.

## 5. Kalamang in-domain ceiling (trained on MTOB's 375 pairs; not transfer)

| Model | Hu prompt, with resources | Hu prompt, task-only | SWELLS-format, with resources | SWELLS-format, task-only |
|---|---|---|---|---|
| Kalamang SFT (no-CoT, 4 ep.) | 35.9 / 35.7 (avg 35.8) | 17.2 / 15.7 | 38.2 / 38.1 (avg 38.1) | 17.7 / 16.3 |
| Kalamang SFT → GRPO (chrF, 1 ep.) | **39.8 / 37.4** (avg 38.6) | 14.9 / 12.6 | **41.3 / 42.5** (avg 41.9) | not run |

Caveat: both Kalamang-trained models never emit EOS (repeat the translation tag until the cap); the evaluator uses the last tag, identical to the first in 81–85/100 items.

### Pilot A notes (1 Oct)
- Training: 370/370 steps, no crash; rollout reward 0.86→0.93, rule 0.88→0.94, exact-match rate 55%→60–69% (plateau from ~step 100). 236 complete groups: 32 had 0/8 exact, 66 had 8/8.
- Partition-4 test (unseen items): 31 / 30 vs SFT 28 / 31 → no significant change (n=100/direction, ±~4.5 SE). One course of RL sharpens the training items but does not move the held-out phenomenon.
- Failure modes on p4: art→eng = tense/aspect errors ("will" vs "would", "were watching" vs "had watched"); eng→art = mostly whole-constituent word-order inversions (object–verb–subject) with correct morphology, plus a few repetition loops.
- **EOS finding (pre-existing, from SFT, not RL)**: `sft_train.py` sets `pad_token = eos_token`, so the EOS after the answer is masked as padding and the SFT model never learns to stop. SFT W+G outputs: 1,109/1,400 hit the 4,096-token cap with repeated garbage after the `<translation>`, 760 contain >1 tag. Pilot A inherits it (58–69/100 at cap); in GRPO all 8 rollouts were clipped at 1,024 tokens for the first ~180 steps (clipped_ratio 1.0 → 0.125 → 0). Accuracy is unaffected (first tag is scored) but eval is ~5× slower and RL spends its budget on post-answer tokens. Fix for a rerun: use a distinct pad token / keep EOS in labels.
- Full W+G regression eval of Pilot A vs SFT running (`eval/grpo_v2_pilot_course4_cot_sftinit_wg_4k`).

## 1c. New W+G evals landed 1 Oct (official acc / chrF, art→eng | eng→art; raw in results/raw/*_summary.json)

| Model | art→eng acc | chrF | eng→art acc | chrF |
|---|---|---|---|---|
| Qwen3-8B instruct, thinking ON (16k ctx, untrained) | 35.5 | 72.2 | 22.3 | 74.5 |
| 4B GRPO-from-base, 2-epoch final stage (CoT prompt) | 42.4 | 80.9 | 6.1 | 68.4 |
| 4B GRPO-from-base, 1 epoch, no-CoT prompt | 15.9 | 65.3 | 2.9 | 54.6 |
| 4B GRPO-from-SFT, 1 epoch, no-CoT prompt | 13.5 | 63.5 | 2.6 | 57.6 |

Notes: thinking-on adds +8/+10 acc over thinking-off (27.6/11.9). Prediction-level inspection confirms that the
2-epoch GRPO-from-base art→eng jump (13.3 → 42.4, above curriculum SFT's 41.2) is genuine, not a tag-extraction
artifact: all 1,400 outputs are short direct answers with exactly one `<translation>` tag. The gain is concentrated
in partitions 1, 2f/2m, 6, and 7f/7m (52–100%); partitions 3–5 remain 11–37%, and eng→art remains much worse than
SFT (6.1 vs 23.6).
| Pilot A (v2+CoT GRPO from 4B SFT, course 4) — full W+G regression | 40.9 | 73.5 | 22.2 | 70.1 |

Pilot A full W+G ≈ curriculum SFT (41.2/73.3 | 23.6/71.8): one course of v2+CoT RL neither helped nor hurt the other partitions.
| 4B GRPO-from-SFT, 2-epoch final stage (CoT prompt) | 40.1 | 73.9 | 20.3 | 73.7 |
| **Qwen3-8B instruct, curriculum SFT (CoT targets, same frozen items)** | **50.3** | **80.7** | **33.9** | **78.5** |
| Qwen3-8B instruct, CoT chrF-GRPO after courses 1–7 (before final stage; thinking off, 4k) | 34.4 | 55.0 | 23.8 | 63.3 |
| Qwen3-8B instruct, CoT chrF-GRPO after courses 1–7 (thinking ON, 16k gen) | 37.8 | 75.3 | 25.7 | 78.0 |
| Qwen3-8B instruct, CoT chrF-GRPO final stage, epoch 1 (step 1410; thinking ON, 16k gen; raw: results/raw/grpo8b_ep1_wg_think16k/) | 43.4 | 77.9 | 27.3 | 80.9 |
| **Qwen3-8B instruct, CoT chrF-GRPO final stage, 2 epochs (step 2816; thinking ON, 4k gen; raw: results/raw/grpo8b_cot_final_wg_4k/)** | **52.6** | **83.9** | 26.4 | **83.5** |

8B CoT chrF-GRPO final (2 final-stage epochs, landed 7 Oct): art→eng 52.6 / 83.9 beats 8B SFT (50.3 / 80.7) — first local RL arm above SFT —
and keeps Latin (8a/8g/9a/9g 47/11/40/13 vs SFT ≈ 0); eng→art 26.4 / 83.5 stays below SFT exact (33.9) but above it on chrF (78.5).
Per-partition acc art→eng | eng→art: 1: 96|88, 2m: 99|52, 2f: 98|36, 3: 71|61, 4: 61|44, 5m/5f: 58/42|13/15, 6: 35|6, 7m/7f: 31/34|1/0,
Latin 47/11/40/13 | 47/0/7/0. Not truncation (6 / 8 of 1,400 without a tag, median 3.1k / 3.2k chars): eng→art fails on producing
agreement/relative-clause morphology while stems are right, so chrF stays high — the chrF blind spot. Epoch 1 → 2: 43.4 → 52.6 | 27.3 → 26.4.
Qualitative per-model analysis (all SWELLS and Hus arms): results/ERROR_ANALYSIS.md.

8B CoT chrF-GRPO course-7 checkpoint (landed 5 Oct, raw: results/raw/grpo8b_course7_wg/): vs the matched untrained
thinking-off row (27.6/11.9) exact acc rises to 34.4/23.8 — eng→art doubles — but stays below 8B SFT (50.3/33.9).
The chrF drop (70.5→55.0 / 70.7→63.3) is a length artifact: 365/1400 art→eng and 213/1400 eng→art outputs hit the
4k cap inside the reasoning and never emit a tag (scored on the truncated reasoning). Per-partition acc art→eng|eng→art:
1: 99|85, 2f: 64|36, 2m: 85|53, 3: 58|63, 4: 27|26, 5f/5m: 24/27|8/10, 6: 10|1, 7f/7m: 3/7|0/0, Latin 8a/8g/9a/9g:
46/8/18/5 | 43/0/7/0. Unlike 8B SFT (Latin ≈0), GRPO keeps the instruct model's Latin generalization (untrained 47/13/25/12 | 40/1/6/0).
Final-stage (2-epoch) checkpoint gets the same eval when cot2 finishes.

8B course-7 checkpoint re-evaluated with thinking ON and a 16k generation cap (landed 5 Oct, raw: results/raw/grpo8b_course7_wg_think16k/):
37.8 / 75.3 | 25.7 / 78.0, vs the matched untrained thinking-on row 35.5 / 72.2 | 22.3 / 74.5 and 8B SFT 50.3 | 33.9. The 4k chrF
artifact is gone (chrF back above untrained); exact accuracy is +2 / +3 over untrained. 73 / 58 of 1,400 outputs still reason past
16k without an answer. Per-partition acc (art→eng | eng→art): 1: 97|84, 2f: 83|42, 2m: 92|58, 3: 55|63, 4: 25|29, 5f/5m: 25/23|8/12,
6: 12|0–3, 7f/7m: 7/8|0/0, Latin 8a/8g/9a/9g: 47/13/29/13 | 49/0/13/0 — the untrained profile with Latin preserved; partitions 4–7
(where SFT reaches 40–78 | 34–43) remain unlearnt. Rollout-level explanation (reward variance per partition): paper/NOTES_AND_PLAN.md, 5 Oct.

8B curriculum SFT (landed 2 Oct, raw: results/raw/sft8b_instruct_wg_4k_summary.json) is the best trained model so far in both
directions: +9/+10 acc over 4B SFT (41.2/23.6) and +15/+12 over the untrained 8B thinking-on baseline (35.5/22.3).
Per-partition acc (art→eng | eng→art): 1: 97|78, 2f: 100|43, 2m: 97|55, 3: 63|63, 4: 46|43, 5f: 42|40, 5m: 41|36,
6: 61–69|35–42, 7f: 78|34, 7m: 67|42, Latin 8a/8g/9a/9g: 6/3/0/1 | 1/0/0/0 (chrF 57–63 | 54–57).
Same pattern as 4B SFT: large gains on trained French-derived families, Latin held-out still ≈0 exact. Unlike 4B SFT the
8B SFT outputs terminate (only 2/1400 hit the 4k cap; 5/3 missing tag), so the EOS problem is 4B-specific or less severe here.

GRPO-from-SFT ep2 is back near SFT level (ep1 was 13.2/2.2); Latin still 0–6. The extracted-answer recovery is real,
but 1,399/1,400 art→eng generations exceed 1,000 characters (median 12.5k) because the model reverted to long CoT
and inherited the EOS bug. This is distinct from the clean direct-output GRPO-from-base ep2 result.

## 1d. Held-out Latin: GRPO generalizes better than SFT, but weakly and asymmetrically

Partitions 8a/8g/9a/9g contain rule families absent from training. Scores below are official accuracy / chrF.

| Model | art→eng 8a | 8g | 9a | 9g | eng→art 8a | 8g | 9a | 9g |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B Base | 0/36 | 1/41 | 0/42 | 0/43 | 0/51 | 1/61 | 0/45 | 0/51 |
| Curriculum SFT | 8/56 | 3/57 | 3/56 | 2/54 | 1/53 | 0/54 | 0/52 | 0/49 |
| GRPO-from-base, 2-epoch final | **11/69** | **7/73** | **7/69** | **8/80** | 0/53 | 0/60 | 0/56 | 0/58 |
| GRPO-from-SFT, 2-epoch final | 6/58 | 0/61 | 1/61 | 1/69 | 0/53 | 0/66 | 0/55 | 0/60 |
| 8B instruct, thinking on, untrained | 47/76 | 13/55 | 25/75 | 12/69 | 40/69 | 1/56 | 6/63 | 0/56 |
| 8B instruct, curriculum SFT | 6/63 | 3/60 | 0/59 | 1/57 | 1/57 | 0/57 | 0/57 | 0/54 |
| 8B instruct, CoT chrF-GRPO, courses 1–7 (5 Oct) | 46/73 | 8/33 | 18/40 | 5/25 | 43/60 | 0/32 | 7/42 | 0/19 |
| Gemini 3.1 Pro | 100/99 | 98/98 | 100/100 | 100/100 | 97/97 | 95/96 | 92/94 | 94/95 |

Interpretation: GRPO-from-base is the better Latin generalizer than SFT. On art→eng, its mean exact accuracy is
8.25% versus SFT's 4.0%, and its chrF advantage is 13–26 points. On eng→art, exact accuracy is still zero, but mean
chrF is about five points above SFT (56.8 vs 52.0). Thus GRPO transfers partial translation skill and produces fully
correct unseen-rule outputs about twice as often in one direction; it does not yet induce the rules systematically
(<1 in 10 exact art→eng, essentially zero exact eng→art). The paper should state both findings: **GRPO generalizes
better than SFT to unseen rule families**, while **neither method solves reliable compositional rule application**.

**SFT destroys the instruct model's own Latin generalization (8B, 2 Oct).** Untrained Qwen3-8B instruct with thinking
reads the Latin grammar better than any fine-tuned open model: 47/13/25/12 exact art→eng and 40/1/6/0 eng→art. After the
identical curriculum SFT it falls to 6/3/0/1 and 1/0/0/0 while the trained partitions rise by ~15/+12 points overall
(50.3/33.9 vs 35.5/22.3). Curriculum SFT therefore buys in-distribution accuracy by overwriting the model's own
reasoning with the SWELLS template CoT; this is the forgetting pattern predicted in the notes and makes the
8B CoT chrF-GRPO arm (which only reinforces the model's own rollouts) the decisive comparison.


## 6. Resource controls (added 5 Oct): is the grammar used, or is the cipher decrypted?
Same 1,400 SWELLS CoT test items per direction; chrF / official exact accuracy. Raw: `results/raw/instruct8b_{w,task}/`, `results/raw/sft8b_think_w/`, `results/raw/sft8b_think_task_16k/`, `results/raw/sft4b_w_task_4k/`, `results/gemini31pro_{wonly,wb,wg}_4k/`.

| Model | dir | task-only | W (dictionary) | W+B (+bitexts) | W+G (+grammar) |
|---|---|---|---|---|---|
| Qwen3-8B instruct, untrained, thinking off, 4k | art→eng | 28.4 / 2.7 | 55.3 / 2.6 | — | 70.5 / 27.6 |
| Qwen3-8B instruct, untrained, thinking off, 4k | eng→art | 19.4 / 0.0 | 49.4 / 0.0 | — | 70.7 / 11.9 |
| Qwen3-8B curriculum SFT, thinking on, 16k | art→eng | 23.4 / 0.0 | 60.9 / 8.9 | — | 80.7 / 50.3 |
| Qwen3-8B curriculum SFT, thinking on, 16k | eng→art | 20.0 / 0.0 | 47.8 / 0.0 | — | 78.5 / 33.9 |
| Qwen3-4B curriculum SFT, 4k | art→eng | 23.4 / 0.1 | 58.6 / 6.4 | — | 73.3 / 41.2 |
| Qwen3-4B curriculum SFT, 4k | eng→art | 17.6 / 0.0 | 46.2 / 0.0 | — | 71.8 / 23.6 |
| Gemini 3.1 Pro, thinking uncapped | art→eng | — | 54.9 / 15.1 | 94.9 / 90.4 | 96.5 / 96.7 |
| Gemini 3.1 Pro, thinking uncapped | eng→art | — | 39.4 / 6.5 | 73.3 / 30.1 | 96.7 / 96.4 |

- No decryption: task-only 0% exact (8B). Dictionary alone: word overlap, ~0 exact. Grammar is what makes sentences exact; the effect is largest eng→art (Gemini W-only 6.5% → W+B 30% → W+G 96% exact; 8B W-only 0% → W+G 12%).
- Trained SFT models without the grammar (W-only): 8B 50.3 → 8.9 / 33.9 → 0.0 exact, 4B 41.2 → 6.4 / 23.6 → 0.0 — both fall to untrained level; the residual 6–9% art→eng is partitions 1–2 only (word-order families a dictionary suffices for). Task-only ≈ noise (chrF 18–23, exact 0). The fine-tuned models still depend on the supplied grammar at test time; they have not memorised the rule families (completed 6 Oct, 1,400/1,400 per cell).
- Gemini W-only costs ~12k thinking tokens/item (vs ~5k W+B, ~4k W+G): without grammar it brute-forces the puzzle. G-only was dropped (no scientific point).
- Correction to earlier notes: `evaluate.py` defaults to `enable_thinking=False`; the 8B controls and the 4k W+G row (`wg_4k/instruct8b_summary.json`) are both thinking-off, so they are protocol-matched.

## 7. Real languages (Hus & Anastasopoulos 2024 archive, private; 18 langs × 2 dirs × 100 items; sentence chrF)
Table: `results/HUS_B2S_QWEN4B_BASE_VS_GEMINI31PRO.tsv` (book+dict / dict / none).
### 7.1 Real-language SFT (Qwen3-4B-Base, Mizo+Ilokano+Kimbundu, book+dict prompts, 3 epochs on all 5,982 train prompts)
Table: `results/HUS_B2S_QWEN4B_SFT_MIK.tsv` (same evaluator, 10,002 items, 0 errors, plain completion, greedy).
Mean chrF over 18 languages, book+dict / dict / none: X→En **31.9 / 32.1 / 31.2** (untrained 25.8 / 24.9 / 28.3); En→X **18.1 / 18.2 / 17.1** (untrained 13.2 / 15.2 / 12.0); Gemini 53.9 / 41.9.
- Trained languages En→X (book+dict): Ilokano 15.8 → 48.0, Mizo 12.4 → 29.1, Kimbundu 8.9 → 26.6; and their *none* rows rise just as much (Ilokano 23.0 → 44.0) → the language itself was memorised.
- Unseen languages move +1–5 (Kalamang 20.2 → 23.0, Chokwe 9.5 → 18.7, Arapaho 3.9 → 8.9 En→X); X→En gains are largely output discipline (the base model's long-context failure in the *none*/context rows is gone).
- Resource use did **not** improve: book+dict − dict = −0.2 / −0.1, dict − none = +0.9 / +1.1. Same picture as SWELLS SFT: memorise, not read.
- GRPO-from-base and GRPO-from-SFT (same subset) are evaluated with the same three contexts: §7.2.

### 7.2 Real-language GRPO-from-base (Qwen3-4B-Base, chrF-GRPO, stratified 1,998-prompt subset of the same train set, 1 epoch, 8 rollouts, book+dict prompts)
Table: `results/HUS_B2S_QWEN4B_GRPO_BASE_MIK_SUB2K.tsv` (10,002 items, 0 errors). Protocol deviation vs SFT: 1/3 of the prompts, 1 epoch (full schedule was ~21 h/epoch).
Mean chrF, book+dict / dict / none: X→En **31.9 / 32.4 / 33.3**; En→X **24.4 / 23.9 / 21.0**.
- En→X book+dict, trained languages (untrained / SFT / GRPO): 12.4 / 34.6 / 31.0; the 15 unseen languages: 13.4 / 14.8 / 23.1. GRPO transfers (Kalamang 20 → 28, Chokwe 10 → 20, Waray 23 → 41), SFT memorises.
- Dictionary contribution En→X: untrained +3.1, SFT +1.1, GRPO +2.9. Grammar-book contribution: −1.9 / −0.1 / +0.5 — no model has learnt to use the book yet.
- X→En: all contexts ≈ 32–33, i.e. the GRPO model ignores the resources in that direction too (the *none* row is the best).
- GRPO-from-SFT (same subset, from the merged SFT model): §7.2b.

### 7.2b Real-language GRPO-from-SFT (Qwen3-4B, chrF-GRPO from the merged §7.1 SFT model, same 1,998-prompt subset, 1 epoch, 8 rollouts)
Table: `results/real_language/hus_chrf_grpo4b-sft.tsv` (10,002 items, 0 errors; `hus_chrf_sft4b.tsv`, `hus_chrf_grpo4b-base.tsv` regenerated alongside).
Mean chrF, book+dict / dict / none: X→En **33.9 / 34.1 / 33.7**; En→X **19.2 / 19.3 / 17.5** (SFT 31.9 / 18.1; GRPO-from-base 31.9 / 24.4).
- En→X stays at SFT level: trained languages Ilokano 46.7 / Mizo 31.3 / Kimbundu 26.4 (SFT 48.0 / 29.1 / 26.6); unseen languages do not recover the GRPO-from-base gain (Kalamang 20.4 vs 28, Chokwe 19.0 vs 20, Waray 30.9 vs 41).
- Resource use unchanged: book+dict − dict = −0.2 / −0.1, dict − none = +0.4 / +1.8.
- Same conclusion as SWELLS 4B (41.2/23.6 → 40.1/20.3): RL after SFT polishes the memorised model; the generalisation lost in SFT is not restored. Order matters.

- Gemini 3.1 Pro (10,002 items, 0 errors, ~29.9M tokens): mean X→En 53.9 / 53.4 / 52.8, En→X 41.9 / 40.9 / 40.6. 15/18 languages known (±1). Unseen: Kalamang 17 → 39 → 43 (both dirs); Natugu book +13.5 (X→En) / +9.2 (En→X) over dict; Arapaho dict +3.5, book 0.
- Qwen3-4B-Base untrained (plain completion, greedy, no CoT): X→En 25.8 / 24.9 / 28.3, En→X 13.2 / 15.2 / 12.0. Uses the dictionary En→X on unseen langs (Kalamang 2 → 23, Kachin 1 → 13, Arapaho 3 → 15); grammar excerpts 0 or negative; any context hurts X→En.
- Training now: 4B-Base LoRA SFT on Mizo + Ilokano + Kimbundu (5,982 book+dict prompts, direct targets, 3 epochs), then chrF-GRPO from base and from SFT; eval 18 langs × 3 contexts; transfer pairs Kimbundu→Chokwe, Ilokano→Waray. Latgalian swapped for Mizo because 4B already knows Latgalian (none 35 X→En).

### 7.2c Second model family (Llama-3.2-3B-Instruct SFT) and Hus "+5 retrieved examples" context (7 Oct)

- Llama-3.2-3B-Instruct (ungated `unsloth/` mirror) SFT on the **same 5,982 MIK prompts and schedule as the Qwen SFT** (LoRA, 3 ep, lr 1e-4, max_len 10,240, no example dropped; loss 2.1→1.2). Raw: `results/real_language/hus_outputs_llama3b-sft/`.
- New inference-only context `book_dict_ex` / `dict_ex`: the Hus `--use_reference_sentences` setup — 5 parallel pairs from the language's Hus *train* split, LCS-retrieved per test sentence (item itself, exact source/reference matches and duplicate pairs excluded; max prompt 10.4k tokens). This is the Hanxu/MTOB-style prompt; unlike Hanxu, our models were **not trained with examples** (lower bound). Raw: `results/real_language/hus_examples/<tag>/`. Table: `results/real_language/HUS_EXAMPLES_LLAMA_VS_QWEN.tsv`. Code: `code/real_language/`.
- chrF, mean over languages (trained 3 / unseen 14, Kalamang excluded), en→X / X→en, book+dict → book+dict+5ex:
  - Qwen3-4B-Base: trained 12.4→20.0 / 28.1→30.0; unseen 12.9→19.0 / 25.5→29.2 (examples are the biggest lever for the *untrained* model, +4–8).
  - Qwen3-4B SFT: trained 34.5→35.5 / 34.5→35.5; unseen 14.2→15.9 / 32.0→33.3 (+1–2).
  - Qwen3-4B chrF-GRPO (sub2k): trained 31.0→31.5 / 32.9→32.8; unseen 22.8→25.6 / 32.2→32.9.
  - Llama-3.2-3B-Instruct: trained 16.5→18.7 / 19.0→20.7; unseen 12.6→15.1 / 20.0→20.3.
  - Llama-3.2-3B SFT: trained 35.2→34.6 / 33.3→33.1; unseen 13.3→14.3 / 28.7→29.3 (≈0 to +1).
- Reading: (i) Llama SFT ≈ Qwen SFT on trained languages, worse on unseen X→en (28.7 vs 32.0) — the SFT pattern (memorise trained languages, little transfer) replicates on a second family. (ii) Retrieved examples help trained models little (+0–2) and untrained models a lot; SFT/GRPO gains are therefore not an artefact of our example-free prompt. (iii) book+dict ≈ dict in every cell, with or without examples. (iv) Hanxu-comparable shared-4 unseen (Dinka/Wolof/Guarani/Kachin) en→X with examples: Qwen base 12.9, Qwen GRPO-sub2k 22.8, Llama SFT 12.0 — vs their RL 24.8 / 22.4.

### 7.2d Scoring fix for direct-output arms (8 Oct) and the 8B SWELLS-RL transfer test with its untrained control
`eval_b2s.py` kept only the first non-empty line after the `<translation>` tag; 573/10,002 GRPO-from-base outputs put a bare `<translation>` line first, so they scored as the literal tag. `code/real_language/rescore_hus.py` re-extracts tag-aware (`results/real_language/HUS_RESCORED_TAGFIX.tsv`). Corrected book+dict chrF (en→X / X→en): **GRPO-from-base full: trained 31.7 / 36.1 (was 31.1 / 32.7), unseen 23.5 / 34.8 (was 23.1 / 32.5)**; untrained, SFT and GRPO-from-SFT change by ≤0.1. GRPO-from-base therefore beats SFT on X→en in both groups (36.1 vs 34.5 trained, 34.8 vs 32.1 unseen) and on unseen en→X (23.5 vs 13.9); SFT keeps trained en→X (34.5 vs 31.7).

8B CoT chrF-GRPO final (trained on SWELLS only, never saw a real language) vs the identical untrained Qwen3-8B (thinking on, SWELLS-style CoT prompt, 4k gen), 18 Hus languages, `results/real_language/HUS_SWELLSFMT_8B_COT_GRPO_VS_UNTRAINED.tsv`, book+dict chrF en→X / X→en: unseen-15 **21.9 / 33.6 vs 20.9 / 32.0**, trained-3 22.8 / 35.6 vs 21.1 / 34.9; dict-only unseen 21.7 / 32.5 vs 20.4 / 31.8. Paired bootstrap (1,000 resamples, `HUS_SWELLSFMT_8B_BOOTSTRAP.txt`): unseen X→en +2.1 [+1.0, +3.3] p<0.001; unseen en→X +0.6 [−0.4, +1.6] p=0.14; dict-only +0.7/+0.8 (p 0.07–0.13). Gains concentrate on languages the base model does not know (Kalamang +3/+3, Kachin +3/+4, Dinka +5/+4) and are ≈0 on known ones (Dogri, Mizo, Ilokano, Gitksan). Grammar-book effect (book − dict): Dinka RL +7 / +6 vs untrained 0 / +0.5, Kalamang X→en +3.7 vs +1.1; Natugu +7–8 for both models (base ability). Format: RL 98.8% closed tags vs 92.7%; the untrained model's zero-chrF items are mostly unfinished thinking. Reading: SWELLS RL transfers a small, significant X→en gain and some grammar use on unknown languages; most of the 8B's absolute score is the base model + thinking.

### 7.3 Metric check: chrF vs chrF++ (Hus & Anastasopoulos report chrF++)
`results/HUS_B2S_CHRF_VS_CHRFPP_MEANS.tsv` rescores every Hus output above with sacrebleu chrF++ (word_order=2), the metric of the Back-to-School paper. chrF++ is 2 to 3 points lower than chrF for every model and context (GRPO-from-base En→X 24.4 → 21.1, Gemini 41.9 → 39.1, untrained 13.2 → 11.3); rankings and all resource effects are unchanged. In like-for-like chrF++, 4B GRPO-from-base En→X (21) sits at the level of Hus's GPT-4-turbo zero-shot baseline and well below his W+S prompt (dictionary + retrieved parallel sentences, which we do not use); Gemini (39) is above his best. Hus's paper has no training: it prompts GPT-4-turbo and finds that W+S is the best prompt for 12/16 languages En→X, that adding the full grammar book usually lowers the score, and that books help only for extremely low-resource languages. Our "book ≈ 0 after SFT/GRPO" is consistent with that. Not to be confused with Hanxu Hu et al. (arXiv 2606.06428, chrF-GRPO on Romansh, Qwen3-4B-Base): that paper does train with RL and reports the same two facts we find here — RL beats SFT on unseen languages (zero-shot Kalamang 25.6 → 34.6 vs SFT 28.6) and the grammar contributes minimally in ablation (−0.8 Romansh / −1.5 Kalamang vs −8 for the dictionary).
