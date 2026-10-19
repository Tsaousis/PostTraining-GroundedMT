"""Render ours-vs-Marmonier accuracy tables from the rescored per-file JSONs.

Usage: python make_tables.py results/scores > results/TABLES.md
Each scores/<model>.json is produced by code/rescore.py on an eval dump.
"""

import json
import os
import sys

PARTITIONS = ["1", "2m", "2f", "3", "4", "5m", "5f", "6", "7m", "7f", "8a", "8g", "9a", "9g"]

# Marmonier et al., Tables 3-6, CoT columns only (the configuration we run).
PAPER = {
    ("eng_art", "dictionary_only"): {
        "base": [0.0] * 14,
        "ft": [0.0] * 14,
    },
    ("eng_art", "incidental_bitexts"): {
        "base": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "ft": [7.0, 5.0, 5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    },
    ("eng_art", "grammar_excerpts"): {
        "base": [66.0, 31.0, 3.0, 20.0, 8.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        "ft": [98.0, 95.0, 91.0, 76.0, 91.0, 86.0, 80.0, 90.0, 90.0, 80.0, 3.0, 0.0, 0.0, 0.0],
    },
    ("art_eng", "dictionary_only"): {
        "base": [1.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 7.0, 4.0],
        "ft": [16.0, 21.0, 36.0, 2.0, 4.0, 11.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
    },
    ("art_eng", "incidental_bitexts"): {
        "base": [68.0, 59.0, 72.0, 23.0, 11.0, 7.0, 12.0, 0.0, 2.0, 2.0, 18.0, 18.0, 17.0, 10.0],
        "ft": [85.0, 97.0, 97.0, 18.0, 17.0, 16.0, 15.0, 9.0, 6.0, 9.0, 7.0, 4.0, 0.0, 0.0],
    },
    ("art_eng", "grammar_excerpts"): {
        "base": [100.0, 96.0, 94.0, 33.0, 27.0, 24.0, 16.0, 4.0, 4.0, 5.0, 16.0, 9.0, 16.0, 10.0],
        "ft": [100.0, 100.0, 100.0, 84.0, 63.0, 71.0, 64.0, 93.0, 88.0, 91.0, 4.0, 1.0, 0.0, 1.0],
    },
}
PAPER_AVG = {
    ("eng_art", "dictionary_only"): (0.0, 0.0),
    ("eng_art", "incidental_bitexts"): (1.8, 1.4),
    ("eng_art", "grammar_excerpts"): (9.5, 62.9),
    ("art_eng", "dictionary_only"): (2.4, 7.1),
    ("art_eng", "incidental_bitexts"): (22.8, 27.1),
    ("art_eng", "grammar_excerpts"): (32.4, 61.4),
}
CONDITION = {
    "dictionary_only": "W (dictionary only)",
    "incidental_bitexts": "W+IB (incidental bitexts)",
    "grammar_excerpts": "W+G (grammar excerpts)",
}
MODELS = ["base", "sft", "grpo"]


def key_of(filename):
    direction = "eng_art" if "eng_art" in filename else "art_eng"
    for cond in CONDITION:
        if filename.endswith(cond):
            return direction, cond
    raise ValueError(filename)


def cell(value):
    return "–" if value is None else f"{value:.1f}"


def main(scores_dir):
    scores = {}
    for model in MODELS:
        path = os.path.join(scores_dir, f"{model}.json")
        if os.path.exists(path):
            scores[model] = {key_of(k): v for k, v in json.load(open(path)).items()}

    print("# SWELLS: Qwen3-4B vs Marmonier et al.\n")
    print("Accuracy as defined in the paper (normalised exact match, 50% credit for an")
    print("adjective-order-only error, bag-of-words for the Latin partitions 8/9),")
    print("all CoT, n=100 per partition.\n")

    for direction in ("eng_art", "art_eng"):
        for cond in ("dictionary_only", "incidental_bitexts", "grammar_excerpts"):
            key = (direction, cond)
            if not any(key in s for s in scores.values()):
                continue
            arrow = "eng→art" if direction == "eng_art" else "art→eng"
            print(f"\n## {arrow}, {CONDITION[cond]}, CoT\n")
            print("| partition | ours base | ours SFT | ours GRPO | Marmonier base | Marmonier FT |")
            print("|---|---|---|---|---|---|")
            for i, part in enumerate(PARTITIONS):
                ours = [
                    cell(scores.get(m, {}).get(key, {}).get("by_partition", {}).get(part))
                    for m in MODELS
                ]
                print(
                    f"| {part} | {ours[0]} | {ours[1]} | {ours[2]} | "
                    f"{PAPER[key]['base'][i]:.1f} | {PAPER[key]['ft'][i]:.1f} |"
                )
            avgs = [cell(scores.get(m, {}).get(key, {}).get("average")) for m in MODELS]
            lat = [cell(scores.get(m, {}).get(key, {}).get("latin_8_9")) for m in MODELS]
            pb, pf = PAPER_AVG[key]
            paper_latin_b = sum(PAPER[key]["base"][10:]) / 4
            paper_latin_f = sum(PAPER[key]["ft"][10:]) / 4
            print(f"| **average** | **{avgs[0]}** | **{avgs[1]}** | **{avgs[2]}** | "
                  f"**{pb:.1f}** | **{pf:.1f}** |")
            print(f"| **Latin 8+9** | {lat[0]} | {lat[1]} | {lat[2]} | "
                  f"{paper_latin_b:.1f} | {paper_latin_f:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
