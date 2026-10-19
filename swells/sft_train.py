"""SFT baseline for SWELLS on Qwen3-4B.

Loss is taken on the assistant turn only; the prompt (which carries the grammar
book and dictionary, and is by far the bulk of the sequence) is masked out.
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

IGNORE = -100

FALLBACK_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


class SequentialTrainer(Trainer):
    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)


class SwellsSFT(Dataset):
    def __init__(self, path, tokenizer, max_len, disable_thinking=False, fail_on_skip=False):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = []
        skipped = 0
        with open(path) as f:
            for line in f:
                msgs = json.loads(line)["messages"]
                prompt_ids = tokenizer.apply_chat_template(
                    msgs[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=not disable_thinking,
                )
                full_ids = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=True,
                    enable_thinking=not disable_thinking,
                )
                if len(full_ids) > max_len:
                    if fail_on_skip:
                        raise ValueError(
                            f"{path}: example of {len(full_ids)} tokens exceeds max_len {max_len}"
                        )
                    skipped += 1
                    continue
                labels = list(full_ids)
                labels[: len(prompt_ids)] = [IGNORE] * len(prompt_ids)
                self.rows.append((full_ids, labels))
        print(f"{path}: kept {len(self.rows)}, skipped {skipped} over {max_len} tokens")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [IGNORE] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--train_file", default="data/sft_train.jsonl")
    ap.add_argument("--output_dir", default="runs/sft")
    ap.add_argument("--max_len", type=int, default=10240)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--no_liger", action="store_true")
    ap.add_argument("--disable_thinking", action="store_true")
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        help="consume the file in order (pre-shuffled dropout curricula)",
    )
    ap.add_argument(
        "--fail_on_skip",
        action="store_true",
        help="abort instead of silently skipping examples over --max_len",
    )
    args = ap.parse_args()

    # Qwen3's 151k vocab makes the logits tensor the memory bottleneck at 10k
    # context (~6 GB per micro-batch on its own); Liger's fused linear+CE never
    # materialises it.

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = FALLBACK_TEMPLATE

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
    except (ImportError, ValueError) as e:
        print(f"flash-attn unavailable ({e}); falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    model.config.use_cache = False

    if not args.no_liger:
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=model)
            print("liger fused linear cross-entropy enabled")
        except Exception as e:
            print(f"liger unavailable ({e}); standard CE, expect higher memory")

    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
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
            ),
        )
        model.print_trainable_parameters()

    ds = SwellsSFT(
        args.train_file,
        tok,
        args.max_len,
        disable_thinking=args.disable_thinking,
        fail_on_skip=args.fail_on_skip,
    )

    trainer_cls = SequentialTrainer if args.no_shuffle else Trainer
    trainer = trainer_cls(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=8,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False,
            report_to=[],
        ),
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()
    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tok.save_pretrained(final)

    if args.lora and trainer.is_world_process_zero():
        merged_dir = os.path.join(args.output_dir, "merged")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"merged model written to {merged_dir}")


if __name__ == "__main__":
    main()
