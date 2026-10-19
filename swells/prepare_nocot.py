"""Build a no-CoT SFT training file from the CoT training file.

Swaps the CoT instruction in the user turn for the no-CoT instruction used by
Test_no_CoT, and reduces the assistant turn to the <translation> tag alone.
"""
import argparse
import json
import re

COT_INSTR = (
    "You may explain your chain of thoughts prior to producing the required "
    "translation."
)
NOCOT_INSTR = (
    "Do not explain your chain of thoughts, instead immediately produce the "
    "required translation in a spontaneous fashion."
)
TAG = re.compile(r"<translation>(.*?)</translation>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default="data/sft_train.jsonl")
    ap.add_argument("--out_file", default="data/sft_train_nocot.jsonl")
    args = ap.parse_args()

    kept = 0
    dropped = 0
    with open(args.in_file) as fin, open(args.out_file, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            msgs = rec["messages"]
            assistant = msgs[-1]["content"]
            m = TAG.search(assistant)
            if not m:
                dropped += 1
                continue
            out_msgs = []
            for msg in msgs[:-1]:
                content = msg["content"]
                if msg["role"] == "user":
                    if COT_INSTR in content:
                        content = content.replace(COT_INSTR, NOCOT_INSTR)
                    elif NOCOT_INSTR not in content:
                        dropped += 1
                        out_msgs = []
                        break
                out_msgs.append({"role": msg["role"], "content": content})
            if not out_msgs:
                continue
            out_msgs.append(
                {
                    "role": "assistant",
                    "content": "<translation>%s</translation>" % m.group(1).strip(),
                }
            )
            fout.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
            kept += 1
    print("kept", kept, "dropped", dropped)


if __name__ == "__main__":
    main()
