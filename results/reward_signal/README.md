# Reward-signal check: can a morphology-aware reward help chrF-GRPO? (4B, course-7 model)

Question: before spending a full final-stage run, does 0.5·chrF + 0.5·exact (or word-exact)
give GRPO a different learning signal than chrF alone?

Setup (`code/reward_signal_check.py`): Qwen3-4B-Base chrF-GRPO course-7 merged model, 200 random
final-stage prompts (`final_1408.jsonl`, no-CoT), 8 rollouts each at temperature 1.0, cap 192 tokens
— identical to what the GRPO trainer samples. Each rollout scored with chrF, sentence exact, word-exact
(fraction of position-aligned words exactly right).

| metric | all (200) | eng→art (180) | art→eng (20) |
|---|---|---|---|
| mean chrF of rollouts | 0.731 | 0.724 | 0.794 |
| mean word-exact | 0.177 | 0.138 | 0.522 |
| groups with ≥1 exact rollout | 11.5% | 7% | 50% |
| groups with all 8 exact | 7% | 4% | 30% |
| distinct outputs per group of 8 | — | 3.2 | 2.5 |
| mean chrF spread within group | 0.050 | | |
| argmax rollout changes, chrF → chrF+exact | 1.0% | | |
| argmax rollout changes, chrF → chrF+word-exact | 10% | 10% | 10% |
| groups where a correct rollout is within 0.05 chrF of a wrong one | 2.5% | | |

Reading:
- Sentence-exact as a reward term is useless here: on eng→art only 7% of groups contain any exact
  rollout, and in 4% all 8 are exact (no contrast) → it changes the group ranking in 1% of groups.
- Word-exact exposes the suffix problem (0.14 vs chrF 0.72: roughly one word in seven is fully right)
  and changes the ranking in ~10% of groups — a real but small extra signal.
- The dominant limitation is rollout collapse: after 7 courses the policy produces ~3 distinct outputs
  per 8 samples with chrF spread 0.05, so GRPO has little to contrast regardless of the reward.
  (The chrF run's own log agrees: `frac_reward_zero_std` 0.28–0.44 over final-stage steps 1–100,
  `chrf_run_final_steps1-120.json`.)

Follow-up: 100-step pilot with REWARD=chrf_wordexact from the same course-7 model
(`runs/pilot_wordexact_c7`), compared with steps 1–100 of the chrF run (exact 0.115 / 0.190 / 0.040 /
0.130 over 25-step windows). Files: `c7_wordexact_signal.json` (all groups + rollouts).

## Finer graders, rescored offline on the same 1,600 rollouts (`code/rescore_signal_graders.py` → `c7_finer_graders.json`)

| grader | mean | within-group spread | groups with zero spread | argmax ≠ chrF (0.5 mix) | correct − wrong separation* |
|---|---|---|---|---|---|
| chrF | 0.731 | 0.050 | 31.5% | — | 0.10 |
| sentence exact | 0.094 | 0.045 | 95.5% | 1% | 1.00 |
| word-exact | 0.177 | 0.082 | 79.5% | 10% | 0.79 |
| per-word edit similarity (position-aligned) | 0.336 | 0.112 | 30% | 32% | 0.64 |
| aligned word similarity (word-level edit distance, letter-cost substitutions) | 0.424 | 0.090 | 29.5% | 28.5% | 0.34 |
| bag-of-words exact (order-free) | 0.468 | 0.125 | 54% | 12.5% | 0.45 |
| word-order score | 0.348 | 0.262 | 67% | 10.5% | 0.69 |
| ending agreement (same stem → same suffix) | 0.276 | 0.173 | 80% | 10% | 0.69 |

*mean score of exact rollouts minus mean of wrong rollouts, in the 9 groups that contain both.

Reading: the finest grader (per-word edit similarity) is the only one that is dense everywhere chrF is
(zero-spread groups 30% vs 31.5%) **and** re-ranks a third of groups while separating correct from
near-miss 6× more strongly than chrF (0.64 vs 0.10). Word-level graders that are exact (word-exact, ending)
are morphology-sensitive but silent in 80% of groups. 31.5% of groups are 8 identical rollouts — no grader
can help there; that is the rollout-collapse ceiling.

## 100-step direct pilot, REWARD=chrf_wordexact (0.5·chrF + 0.5·word-exact), 4B course-7, same protocol as main arm

`pilot_direct_wordexact_c7/trainer_state_step100.json`. In-rollout exact-match (diagnostic, not rewarded in chrF run) per 25-step window:

| steps | chrF run (chrF reward) | pilot (chrF + word-exact) |
|---|---|---|
| 1–25 | 0.115 | 0.120 |
| 26–50 | 0.190 | 0.175 |
| 51–75 | 0.040 | 0.040 |
| 76–100 | 0.130 | 0.20 (76–90) |

Identical curve within noise: the prompts arrive in the same order and 30–50% of groups have zero
reward spread, so the extra term changes nothing GRPO can use. Full 2-epoch chrF+word-exact ablation dropped.
