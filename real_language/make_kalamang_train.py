"""Turn the MTOB-format train prompts (build_mtob_prompts.py) into the
messages-JSONL used by sft_train.py / grpo_train.py: SWELLS system prompt,
MTOB prompt + the SWELLS no-CoT output instruction, and a direct
<translation>...</translation> target. Both directions are interleaved and
shuffled with a fixed seed so SFT and GRPO see the identical file.
"""

import argparse
import json
import random

SYSTEM = "You are an expert linguist and translator."
SUFFIX = (
    "\n\nDo not explain your chain of thoughts, instead immediately produce the "
    "required translation in a spontaneous fashion. IMPORTANT: Do write your "
    "translation between tags in the following manner: "
    "<translation>your translation here</translation>."
)


def rows(path, direction):
    for line in open(path):
        r = json.loads(line)
        yield {
            "direction": direction,
            "id": r["id"],
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["prompt"] + SUFFIX},
                {"role": "assistant", "content": f"<translation>{r['target']}</translation>"},
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = list(rows(f"{args.data_dir}/train_ke.jsonl", "ke")) + list(rows(f"{args.data_dir}/train_ek.jsonl", "ek"))
    random.Random(args.seed).shuffle(data)
    with open(args.out, "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(data), "examples ->", args.out)


if __name__ == "__main__":
    main()
