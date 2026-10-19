"""GRPO on SWELLS with a chrF reward, plus per-rollout logging for the pilot.

The pilot's purpose is diagnostic: `analyze_pilot.py` reads the reward log and
reports what fraction of GRPO groups are flat (all rollouts scoring alike), which
is the failure mode short targets invite.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from swells_reward import compute_score, extract_translation  # noqa: F401

FALLBACK_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_rows(path, limit=None, max_prompt_chars=None):
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            gold = extract_translation(msgs[-1]["content"])
            if not gold:
                continue
            if max_prompt_chars and len(msgs[1]["content"]) > max_prompt_chars:
                continue
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": msgs[1]["content"]},
                    ],
                    "gold": gold,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


# chrF is the frozen main-grid reward; the mixes are labelled ablations.
REWARD_WEIGHTS = {
    "chrf": [1.0, 0.0, 0.0, 0.0],
    "chrf_exact": [0.5, 0.5, 0.0, 0.0],
    "chrf_wordexact": [0.5, 0.0, 0.5, 0.0],
    "chrf_aligned_wordexact": [0.5, 0.0, 0.25, 0.25],
}


def norm(text):
    return (text or "").strip().strip("@").strip().lower().rstrip(".").strip()


def make_reward_fn():
    def chrf_reward(completions, gold, **kwargs):
        scores = []
        for comp, g in zip(completions, gold):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            scores.append(compute_score("swells", text, g))
        return scores

    # Weight 0: logged by TRL as rewards/exact_match_diagnostic so we can see
    # whether chrF gains translate into the paper's exact-match accuracy.
    def exact_match_diagnostic(completions, gold, **kwargs):
        scores = []
        for comp, g in zip(completions, gold):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            scores.append(
                1.0 if norm(extract_translation(text)) == norm(g) else 0.0
            )
        return scores

    # Fraction of gold words reproduced exactly (position-aligned); a wrong
    # affix costs a whole word instead of a few characters under chrF.
    def word_exact(completions, gold, **kwargs):
        scores = []
        for comp, g in zip(completions, gold):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            p = norm(extract_translation(text)).split()
            gw = norm(g).split()
            if not p or not gw:
                scores.append(0.0)
                continue
            hits = sum(1 for a, b in zip(p, gw) if a == b)
            scores.append(hits / max(len(gw), len(p)))
        return scores

    def _lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def aligned_word_sim(completions, gold, **kwargs):
        # word-level edit distance; substituting a word costs its normalised letter distance
        scores = []
        for comp, g in zip(completions, gold):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            p = norm(extract_translation(text)).split()
            gw = norm(g).split()
            if not p or not gw:
                scores.append(0.0)
                continue
            prev = [float(j) for j in range(len(gw) + 1)]
            for i, a in enumerate(p, 1):
                cur = [float(i)]
                for j, b in enumerate(gw, 1):
                    sub = prev[j - 1] + _lev(a, b) / max(len(a), len(b), 1)
                    cur.append(min(prev[j] + 1, cur[j - 1] + 1, sub))
                prev = cur
            scores.append(max(0.0, 1 - prev[-1] / max(len(p), len(gw))))
        return scores

    return [chrf_reward, exact_match_diagnostic, word_exact, aligned_word_sim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--train_file", default="data/sft_train.jsonl")
    ap.add_argument("--output_dir", default="runs/pilot")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_prompt_chars", type=int, default=None)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_prompt_length", type=int, default=12288)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--vllm_gpu_util", type=float, default=0.3)
    ap.add_argument("--vllm_mode", default="colocate", choices=["colocate", "server"])
    ap.add_argument("--no_vllm", action="store_true")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--epsilon_high", type=float, default=None)
    ap.add_argument("--liger_loss", action="store_true")
    ap.add_argument("--vllm_server_port", type=int, default=8000)
    ap.add_argument("--resume_from_checkpoint", default=None)
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        help="consume the file in order (pre-shuffled dropout curricula)",
    )
    ap.add_argument("--disable_thinking", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault(
        "SWELLS_REWARD_LOG", os.path.join(args.output_dir, "rewards.jsonl")
    )

    rows = load_rows(args.train_file, args.limit, args.max_prompt_chars)
    print(f"GRPO prompts: {len(rows)}")
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE
    elif args.disable_thinking:
        tok.chat_template = "{% set enable_thinking = false %}" + tok.chat_template

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    extra = {
        "model_init_kwargs": {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    }
    if args.liger_loss:
        extra["use_liger_loss"] = True
    if args.vllm_mode == "server":
        extra["vllm_server_port"] = args.vllm_server_port
        extra["vllm_server_timeout"] = 1200.0
    if args.epsilon_high is not None:
        extra["epsilon_high"] = args.epsilon_high

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=-1 if args.epochs else args.max_steps,
        num_train_epochs=args.epochs if args.epochs else 3.0,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        temperature=args.temperature,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_completions=True,
        num_completions_to_print=2,
        report_to=[],
        use_vllm=not args.no_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        beta=args.beta,
        reward_weights=REWARD_WEIGHTS[os.environ.get("REWARD", "chrf")],
        shuffle_dataset=not args.no_shuffle,
        **extra,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=make_reward_fn(),
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
