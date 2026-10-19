"""Offline reward-signal check: sample GRPO-style rollout groups from a model and
measure how often a mixed reward re-ranks rollouts that chrF treats as near-ties."""
import argparse
import json
import random
import statistics

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from grpo_train import load_rows, make_reward_fn, norm
from swells_reward import extract_translation

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--train_file", required=True)
ap.add_argument("--n_prompts", type=int, default=200)
ap.add_argument("--n_gen", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=192)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = load_rows(args.train_file)
random.Random(args.seed).shuffle(rows)
rows = rows[: args.n_prompts]

tok = AutoTokenizer.from_pretrained(args.model)
texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in rows]
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=16384 + args.max_tokens, seed=args.seed)
sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=args.n_gen, seed=args.seed)
outs = llm.generate(texts, sp)

chrf_fn, exact_fn, wordexact_fn, aligned_fn = make_reward_fn()

def rank_changed(a, b):
    """True if the argmax rollout under reward a differs from reward b (ties broken by index)."""
    return max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__)

records, stats = [], {"n_groups": 0, "chrf_spread": [], "exact_any": 0, "exact_all": 0, "wordexact_spread": [],
                      "rerank_exact": 0, "rerank_wordexact": 0, "near_tie_wrong_suffix": 0, "chrf_mean": [], "exact_rate": [], "wordexact_mean": []}
for r, o in zip(rows, outs):
    comps = [c.text for c in o.outputs]
    gold = [r["gold"]] * len(comps)
    c = chrf_fn(comps, gold); e = exact_fn(comps, gold); w = wordexact_fn(comps, gold); al = aligned_fn(comps, gold)
    mix_e = [0.5 * x + 0.5 * y for x, y in zip(c, e)]
    mix_w = [0.5 * x + 0.5 * y for x, y in zip(c, w)]
    stats["n_groups"] += 1
    stats["chrf_spread"].append(max(c) - min(c))
    stats["wordexact_spread"].append(max(w) - min(w))
    stats["chrf_mean"].append(statistics.mean(c)); stats["exact_rate"].append(statistics.mean(e)); stats["wordexact_mean"].append(statistics.mean(w))
    if any(e): stats["exact_any"] += 1
    if all(e): stats["exact_all"] += 1
    if rank_changed(c, mix_e): stats["rerank_exact"] += 1
    if rank_changed(c, mix_w): stats["rerank_wordexact"] += 1
    # a correct rollout exists and some wrong rollout is within 0.05 chrF of it
    if any(e):
        best_c = max(ci for ci, ei in zip(c, e) if ei)
        if any(ei == 0 and ci >= best_c - 0.05 for ci, ei in zip(c, e)):
            stats["near_tie_wrong_suffix"] += 1
    records.append({"gold": r["gold"], "preds": [norm(extract_translation(t)) for t in comps], "chrf": c, "exact": e, "word_exact": w, "aligned": al, "raw": comps})

n = stats["n_groups"]
summary = {
    "n_groups": n,
    "mean_chrf": statistics.mean(stats["chrf_mean"]),
    "rollout_exact_rate": statistics.mean(stats["exact_rate"]),
    "mean_word_exact": statistics.mean(stats["wordexact_mean"]),
    "groups_with_any_exact": stats["exact_any"] / n,
    "groups_all_exact": stats["exact_all"] / n,
    "mean_chrf_spread_in_group": statistics.mean(stats["chrf_spread"]),
    "mean_wordexact_spread_in_group": statistics.mean(stats["wordexact_spread"]),
    "argmax_changed_chrf_vs_chrf+exact": stats["rerank_exact"] / n,
    "argmax_changed_chrf_vs_chrf+wordexact": stats["rerank_wordexact"] / n,
    "groups_correct_rollout_tied_with_wrong_under_chrf": stats["near_tie_wrong_suffix"] / n,
}
with open(args.out, "w") as f:
    json.dump({"summary": summary, "groups": records}, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, indent=1))
