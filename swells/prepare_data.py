"""Convert the SWELLS release into SFT jsonl + verl-style parquet for GRPO.

SWELLS ships one freshly generated conlang per instance (1760 distinct conlangs in
the shipped training file), so nothing about a specific cipher is reusable across
examples: the only thing a model can learn is how to exploit the dictionary and
grammar excerpts carried in the prompt.

Gold targets:
  eng->art : `ciphered_reference` (the conlang string)
  art->eng : `reference` (English)
For the shipped training file the target is the payload of the <translation> tag
in the assistant turn.
"""

import argparse
import json
import os
import random
import re

import pandas as pd

TRANSLATION_RE = re.compile(r"<translation>(.*?)</translation>", re.S)


def extract_translation(text):
    m = TRANSLATION_RE.findall(text or "")
    if not m:
        return ""
    return m[-1].strip().strip("@").strip()


def load_train(path):
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            gold = extract_translation(msgs[-1]["content"])
            if not gold:
                continue
            rows.append(
                {
                    "system": msgs[0]["content"],
                    "prompt": msgs[1]["content"],
                    "completion": msgs[-1]["content"],
                    "gold": gold,
                }
            )
    return rows


def load_test(path):
    direction = "eng_art" if "eng_art" in os.path.basename(path) else "art_eng"
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            gold = rec["ciphered_reference"] if direction == "eng_art" else rec["reference"]
            gold = (gold or "").strip().strip("@").strip()
            if not gold:
                continue
            rows.append(
                {
                    "system": "You are an expert linguist and translator.",
                    "prompt": rec["prompt"],
                    "completion": "",
                    "gold": gold,
                    "instance_type": rec.get("instance_type", ""),
                    "direction": direction,
                }
            )
    return rows


def to_verl(rows, split, source):
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "data_source": source,
                "prompt": [
                    {"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["prompt"]},
                ],
                "ability": "translation",
                "reward_model": {"style": "rule", "ground_truth": r["gold"]},
                "extra_info": {
                    "split": split,
                    "index": i,
                    "instance_type": r.get("instance_type", ""),
                    "direction": r.get("direction", ""),
                },
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swells_dir", default="~/swells_data/SWELLS_Dataset")
    ap.add_argument("--out_dir", default="./data")
    ap.add_argument("--val_size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train = load_train(os.path.join(args.swells_dir, "Train", "train_example.jsonl"))
    random.shuffle(train)
    print(f"train examples: {len(train)}")

    with open(os.path.join(args.out_dir, "sft_train.jsonl"), "w") as f:
        for r in train:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": r["system"]},
                            {"role": "user", "content": r["prompt"]},
                            {"role": "assistant", "content": r["completion"]},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    to_verl(train, "train", "swells").to_parquet(os.path.join(args.out_dir, "train.parquet"))

    test_dir = os.path.join(args.swells_dir, "Test", "Test_CoT")
    val_pool = []
    for name in sorted(os.listdir(test_dir)):
        if not name.endswith(".jsonl"):
            continue
        rows = load_test(os.path.join(test_dir, name))
        print(f"{name}: {len(rows)}")
        to_verl(rows, "test", "swells").to_parquet(
            os.path.join(args.out_dir, name.replace(".jsonl", ".parquet"))
        )
        val_pool.extend(rows)

    random.shuffle(val_pool)
    to_verl(val_pool[: args.val_size], "val", "swells").to_parquet(
        os.path.join(args.out_dir, "val.parquet")
    )
    print(f"val: {min(args.val_size, len(val_pool))}")


if __name__ == "__main__":
    main()
