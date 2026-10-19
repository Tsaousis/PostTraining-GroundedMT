"""Merge a LoRA adapter into its base model so vLLM can serve it directly."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(
        args.out
    )
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
