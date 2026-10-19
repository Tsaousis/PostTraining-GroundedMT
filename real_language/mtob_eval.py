"""Evaluate a model on MTOB test prompts (built by build_mtob_prompts.py) with
vLLM, scoring corpus chrF the way mtob/baselines/eval.py does: punctuation
stripped from predictions and references, sacrebleu CHRF defaults (chrF2,
character n-grams only), which is what HF `evaluate.load("chrf")` computes.

Prediction = first non-empty line of the generation (or the content of a
<translation> tag if the model emits the SWELLS format).
"""

import argparse
import json
import os
import re
import statistics
import string

import sacrebleu
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

FALLBACK_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
TAG_RE = re.compile(r"<translation>(.*?)</translation>", re.S)
PUNCT = str.maketrans("", "", string.punctuation)


def strip_punctuation(s):
    return s.replace(" =", "").translate(PUNCT)


def extract(gen):
    m = TAG_RE.search(gen)
    if m:
        return m.group(1).strip()
    for line in gen.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--gpu_util", type=float, default=0.8)
    ap.add_argument("--chat", action="store_true", help="wrap the MTOB prompt in the chat template used for SWELLS eval")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    summary = {}
    for path in args.test_files:
        rows = [json.loads(l) for l in open(path)]
        if args.chat:
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": "You are an expert linguist and translator."},
                 {"role": "user", "content": r["prompt"]}],
                tokenize=False, add_generation_prompt=True) for r in rows]
        else:
            texts = [r["prompt"] for r in rows]
        outs = llm.generate(texts, sp)

        records, preds, refs = [], [], []
        for r, o in zip(rows, outs):
            gen = o.outputs[0].text
            pred = extract(gen)
            p, g = strip_punctuation(pred), strip_punctuation(r["ground_truth"])
            preds.append(p)
            refs.append(g)
            records.append({"id": r["id"], "source": r["source"], "gold": r["ground_truth"], "pred": pred,
                            "chrf": sacrebleu.sentence_chrf(p, [g]).score, "gen": gen,
                            "finish": o.outputs[0].finish_reason})
        name = os.path.basename(path).replace(".jsonl", "")
        with open(os.path.join(args.output_dir, f"{name}.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary[name] = {
            "n": len(records),
            "corpus_chrf": sacrebleu.corpus_chrf(preds, [refs]).score,
            "mean_sentence_chrf": statistics.mean(r["chrf"] for r in records),
            "empty": sum(1 for p in preds if not p.strip()),
            "hit_max_tokens": sum(1 for r in records if r["finish"] == "length"),
            "used_translation_tag": sum(1 for r in records if TAG_RE.search(r["gen"])),
        }
        print(name, json.dumps(summary[name]))
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
