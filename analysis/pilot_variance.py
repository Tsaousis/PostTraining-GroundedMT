"""GRPO reward-variance pilot without training.

GRPO normalises rewards inside each group of `n` rollouts for the same prompt.
If all rollouts in a group score identically the advantage is exactly zero and
the prompt contributes no gradient. With SWELLS' ~9-token references chrF may be
near-binary, so this measures the fraction of dead groups *before* paying for a
training run: sample n completions per prompt from the policy, score with chrF,
and report within-group spread.
"""

import argparse
import json
import os
import random
import statistics

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--train_file", default="data/sft_train.jsonl")
    ap.add_argument("--output_dir", default="runs/pilot")
    ap.add_argument("--prompts", type=int, default=192)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=12288)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_util", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    with open(args.train_file) as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            gold = extract_translation(msgs[-1]["content"])
            if gold:
                rows.append((msgs[:-1], gold))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.prompts]
    print(f"pilot on {len(rows)} prompts x {args.n} rollouts")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    prompts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m, _ in rows
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
        enforce_eager=False,
    )
    outs = llm.generate(
        prompts,
        SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    groups = []
    with open(os.path.join(args.output_dir, "rewards.jsonl"), "w") as fh:
        for (msgs, gold), out in zip(rows, outs):
            scores = []
            for cand in out.outputs:
                pred = extract_translation(cand.text)
                if pred:
                    s = sacrebleu.sentence_chrf(pred, [gold]).score / 100.0
                else:
                    s = 0.0
                scores.append(s)
                fh.write(
                    json.dumps(
                        {
                            "score": s,
                            "gold": gold,
                            "pred": pred,
                            "has_tag": "<translation>" in cand.text,
                            "gen_len": len(cand.token_ids),
                        }
                    )
                    + "\n"
                )
            groups.append(scores)

    flat = [s for g in groups for s in g]
    stds = [statistics.pstdev(g) for g in groups]
    dead = sum(1 for s in stds if s < 1e-6)
    summary = {
        "prompts": len(groups),
        "rollouts": len(flat),
        "mean_chrf": statistics.mean(flat),
        "median_chrf": statistics.median(flat),
        "zero_reward_rate": sum(1 for s in flat if s == 0.0) / len(flat),
        "perfect_rate": sum(1 for s in flat if s > 0.999) / len(flat),
        "dead_group_rate": dead / len(groups),
        "mean_group_std": statistics.mean(stds),
        "median_group_std": statistics.median(stds),
        "verdict": "USABLE" if dead / len(groups) < 0.6 else "TOO SPARSE",
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
