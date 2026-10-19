"""Evaluate a checkpoint on the SWELLS test conditions with vLLM.

Reports chrF overall and broken down by phenomenon, which is the axis SWELLS is
actually built for: every instance carries its own freshly generated conlang, so
a per-phenomenon breakdown says which linguistic structures a model can pick up
from the grammar book rather than how well it memorised a cipher.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

import sacrebleu
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from swells_reward import extract_translation

FALLBACK_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def norm(text):
    """Paper's normalisation: lowercased, stripped of full stops."""
    return (text or "").strip().strip("@").strip().lower().rstrip(".").strip()


def accuracy(pred, gold, adj_gold, partition):
    """Marmonier et al. accuracy: exact match 100, adjective-order-only error 50,
    bag-of-words matching for the Latin-derived partitions 8 and 9."""
    p, g = norm(pred), norm(gold)
    if not p:
        return 0.0
    if partition.startswith(("8", "9")):
        return 100.0 if sorted(p.split()) == sorted(g.split()) else 0.0
    if p == g:
        return 100.0
    if adj_gold and p == norm(adj_gold):
        return 50.0
    return 0.0


def load(path):
    direction = "eng_art" if "eng_art" in os.path.basename(path) else "art_eng"
    ciphered = direction == "eng_art"
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            gold = rec["ciphered_reference"] if ciphered else rec["reference"]
            gold = (gold or "").strip().strip("@").strip()
            adj_key = (
                "ciphered_adjective_order_error_reference"
                if ciphered
                else "adjective_order_error_reference"
            )
            if not gold:
                continue
            rows.append(
                {
                    "prompt": rec["prompt"],
                    "gold": gold,
                    "adj_gold": (rec.get(adj_key) or "").strip().strip("@").strip(),
                    "instance_type": rec.get("instance_type", ""),
                    "direction": direction,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--output_dir", default="eval_results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--per_partition",
        type=int,
        default=None,
        help="sample N instances per partition (1., 2m., ... 8a., 9g.) instead of "
        "taking a prefix, so held-out Latin-derived partitions are covered",
    )
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Qwen3 chat models: leave thinking mode on (default off; CoT then comes from the prompt)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["<|im_end|>"])

    summary = {}
    for path in args.test_files:
        rows = load(path)
        if args.per_partition:
            buckets = defaultdict(list)
            for r in rows:
                buckets[r["instance_type"].split(".")[0]].append(r)
            rows = [
                r
                for b in sorted(buckets)
                for r in buckets[b][: args.per_partition]
            ]
        elif args.limit:
            rows = rows[: args.limit]
        texts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": "You are an expert linguist and translator."},
                    {"role": "user", "content": r["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for r in rows
        ]
        outs = llm.generate(texts, sp)

        per_type = defaultdict(list)
        per_partition = defaultdict(list)
        scores = []
        accs = []
        records = []
        for r, o in zip(rows, outs):
            gen = o.outputs[0].text
            pred = extract_translation(gen)
            s = sacrebleu.sentence_chrf(pred, [r["gold"]]).score if pred else 0.0
            # the paper reports 6m/6f merged as a single partition 6
            partition = r["instance_type"].split(".")[0]
            if partition in ("6m", "6f"):
                partition = "6"
            a = accuracy(pred, r["gold"], r["adj_gold"], partition)
            scores.append(s)
            accs.append(a)
            per_type[r["instance_type"].split("__")[0]].append((s, a))
            per_partition[partition].append((s, a))
            records.append(
                {"pred": pred, "gold": r["gold"], "chrf": s, "accuracy": a, "gen": gen}
            )

        name = os.path.basename(path).replace(".jsonl", "")
        with open(os.path.join(args.output_dir, f"{name}.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary[name] = {
            "n": len(scores),
            "chrf": statistics.mean(scores) if scores else 0.0,
            "exact_match": sum(
                1 for rec in records if rec["pred"].strip() == rec["gold"].strip()
            )
            / max(len(records), 1),
            "accuracy": statistics.mean(accs) if accs else 0.0,
            "by_phenomenon": {
                k: {
                    "n": len(v),
                    "chrf": statistics.mean(s for s, _ in v),
                    "accuracy": statistics.mean(a for _, a in v),
                }
                for k, v in sorted(per_type.items())
            },
            "by_partition": {
                k: {
                    "n": len(v),
                    "chrf": statistics.mean(s for s, _ in v),
                    "accuracy": statistics.mean(a for _, a in v),
                }
                for k, v in sorted(per_partition.items())
            },
        }
        print(
            f"{name}: chrF {summary[name]['chrf']:.2f} "
            f"acc {summary[name]['accuracy']:.1f} "
            f"EM {summary[name]['exact_match']:.1%} (n={len(scores)})"
        )

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
