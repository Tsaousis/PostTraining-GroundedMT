"""Build the untruncated (4k-cap) W+G partition table for every evaluated model."""
import json
import os

RUNS = [
    ("base 4B", "~/results/base4b_wg_4k/summary.json"),
    ("base 8B", "~/results/base8b_wg_4k/summary.json"),
    ("Qwen3-8B instruct (thinking off)", "~/results/instruct8b_wg_4k/summary.json"),
    ("curriculum SFT 4B", "~/results/sft_curriculum_wg_4k/summary.json"),
    ("GRPO-from-base 4B (1-ep. final)", "~/results/grpo_base_ep1_wg_4k/summary.json"),
]
PARTITIONS = ["1", "2m", "2f", "3", "4", "5m", "5f", "6", "7m", "7f", "8a", "8g", "9a", "9g"]
DIRECTIONS = [
    ("art\u2192eng", "CoT_full_art_eng_test_grammar_excerpts"),
    ("eng\u2192art", "CoT_full_eng_art_test_grammar_excerpts"),
]


def partition_of(key):
    head = key.split(".")[0]
    if head in ("6m", "6f"):
        return "6"
    return head if head in PARTITIONS else None


def aggregate(block):
    totals = {p: [0, 0.0] for p in PARTITIONS}
    for key, v in block["by_phenomenon"].items():
        p = partition_of(key)
        if p is None:
            continue
        totals[p][0] += v["n"]
        totals[p][1] += v["accuracy"] * v["n"]
    return {p: (acc / n if n else 0.0) for p, (n, acc) in totals.items()}


def main():
    out = ["# SWELLS W+G, untruncated (16k context, 4096-token generation cap)", ""]
    for label, key in DIRECTIONS:
        out.append(f"## {label}")
        out.append("")
        names = [n for n, path in RUNS if os.path.exists(path)]
        out.append("| partition | " + " | ".join(names) + " |")
        out.append("|---|" + "---|" * len(names))
        cols = []
        for _, path in RUNS:
            if not os.path.exists(path):
                continue
            cols.append(json.load(open(path))[key])
        per_run = [aggregate(c) for c in cols]
        for p in PARTITIONS:
            out.append(f"| {p} | " + " | ".join(f"{r[p]:.1f}" for r in per_run) + " |")
        out.append("| **average** | " + " | ".join(f"**{c['accuracy']:.1f}**" for c in cols) + " |")
        out.append("| chrF | " + " | ".join(f"{c['chrf']:.2f}" for c in cols) + " |")
        out.append("| exact match | " + " | ".join(f"{100 * c['exact_match']:.1f}%" for c in cols) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
