"""Format-controlled variants of Hu et al.'s Kalamang test parquets.

Hu's prompt ends with a request for step-by-step reasoning and a \\boxed{}
answer.  Our SWELLS-trained models were trained to answer inside
<translation></translation> tags (SFT: with CoT; GRPO: no CoT), so the
\\boxed{} instruction is out of distribution for them.  These variants keep
Hu's prompt (all resources, same sentence) and only swap the final instruction
for the SWELLS one, so that any remaining gap is capability rather than format.
Also prepends the SWELLS system prompt.
"""

import argparse
import pandas as pd

HU_TAIL = (
    "Please provide your meta-linguistic reasoning step by step, "
    "and put your final translation within \\boxed{}."
)
SWELLS_COT = (
    "You may explain your chain of thoughts prior to producing the required "
    "translation. IMPORTANT: Do write your translation between tags in the "
    "following manner: <translation>your translation here</translation>."
)
SWELLS_NOCOT = (
    "Do not explain your chain of thoughts, instead immediately produce the "
    "required translation in a spontaneous fashion. IMPORTANT: Do write your "
    "translation between tags in the following manner: "
    "<translation>your translation here</translation>."
)
SYSTEM = {"role": "system", "content": "You are an expert linguist and translator."}


def convert(df, tail):
    out = df.copy()
    prompts = []
    n_missing = 0
    for p in out["prompt"]:
        msgs = [dict(m) for m in p]
        content = msgs[-1]["content"]
        if HU_TAIL in content:
            content = content.replace(HU_TAIL, tail)
        else:
            n_missing += 1
            content = content.rstrip() + "\n" + tail
        msgs[-1]["content"] = content
        if msgs[0]["role"] != "system":
            msgs = [SYSTEM] + msgs
        prompts.append(msgs)
    out["prompt"] = prompts
    if n_missing:
        print("warning: Hu tail not found in", n_missing, "prompts; appended")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    for name in ["test_kalamang", "test_kalamang_taskonly"]:
        df = pd.read_parquet(f"{args.data_dir}/{name}.parquet")
        for suffix, tail in [("swellscot", SWELLS_COT), ("swellsnocot", SWELLS_NOCOT)]:
            o = convert(df, tail)
            path = f"{args.data_dir}/{name}_{suffix}.parquet"
            o.to_parquet(path)
            print(path, len(o))


if __name__ == "__main__":
    main()
