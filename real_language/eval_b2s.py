"""Untrained/trained Qwen eval on Hus (Back to School) test prompts, 3 contexts, greedy, chrF."""
import argparse, glob, json, os, re
import sacrebleu
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

FALLBACK = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

def clean(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = re.search(r"<translation>\s*(.*?)\s*(?:</translation>|$)", t, flags=re.S)
    if m: t = m.group(1)
    lines = [l for l in t.splitlines() if l.strip() and not re.fullmatch(r"</?translation>", l.strip())]
    t = lines[0] if lines else ""
    t = re.sub(r"^(English|[A-Z][a-z]+) translation:\s*", "", t.strip())
    return t.strip().strip('"')

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--prompts", default=os.path.expanduser("real_language/data/prompts"))
ap.add_argument("--out", default=os.path.expanduser("real_language/data/outputs"))
ap.add_argument("--max_tokens", type=int, default=256); ap.add_argument("--max_model_len", type=int, default=12288)
ap.add_argument("--gpu_util", type=float, default=0.85); ap.add_argument("--enable_thinking", action="store_true")
ap.add_argument("--completion", action="store_true", help="plain completion, no chat template (for Base models)")
ap.add_argument("--suffix", action="store_true", help="append SWELLS no-CoT <translation> instruction")
ap.add_argument("--cot", action="store_true", help="prompts already carry the SWELLS CoT instruction: no suffix, no blank-line stop, long generation")
a = ap.parse_args()
SUFFIX = ("\n\nDo not explain your chain of thoughts, instead immediately produce the required translation in a "
          "spontaneous fashion. IMPORTANT: Do write your translation between tags in the following manner: "
          "<translation>your translation here</translation>.")
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
if tok.chat_template is None: tok.chat_template = FALLBACK
llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, trust_remote_code=True, enable_prefix_caching=True)
_im_end = tok.convert_tokens_to_ids("<|im_end|>")
_stop_ids = [tok.eos_token_id] + ([_im_end] if isinstance(_im_end, int) and _im_end >= 0 else [])
sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=_stop_ids,
                    stop=["</translation>"] if a.cot else ["\n\n"] if not a.enable_thinking else None,
                    include_stop_str_in_output=True)
os.makedirs(f"{a.out}/{a.tag}", exist_ok=True)
for path in sorted(glob.glob(f"{a.prompts}/*.jsonl")):
    lang = os.path.basename(path)[:-6]; outp = f"{a.out}/{a.tag}/{lang}.jsonl"
    if os.path.exists(outp): continue
    rows = [json.loads(l) for l in open(path)]
    kw = {"enable_thinking": a.enable_thinking} if "Base" not in a.model else {}
    if a.completion:
        texts = [r["prompt"] + (SUFFIX if a.suffix else "") + " " for r in rows]
    else:
        texts = [
            tok.apply_chat_template(
                [
                    {"role": "system", "content": "You are an expert linguist and translator."},
                    {"role": "user", "content": r["prompt"] + (SUFFIX if a.suffix else "")},
                ],
                tokenize=False,
                add_generation_prompt=True,
                **kw,
            )
            for r in rows
        ]
    outs = llm.generate(texts, sp)
    with open(outp, "w") as f:
        for r, o in zip(rows, outs):
            raw = o.outputs[0].text; hyp = clean(raw)
            f.write(json.dumps({**{k: r[k] for k in ("lang","direction","idx","original_id","context","source","reference")},
                     "raw": raw, "hyp": hyp, "has_tag": "</translation>" in raw, "chrf": sacrebleu.sentence_chrf(hyp, [r["reference"]]).score}, ensure_ascii=False) + "\n")
    print("DONE", lang, flush=True)
print("ALL_DONE", flush=True); os._exit(0)
