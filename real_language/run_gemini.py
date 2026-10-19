import sys
"""Run Gemini on Hus test prompts (book_dict / dict / none), save raw responses, score chrF. Resumable."""
import json, os, sys, time, glob, concurrent.futures as cf
from google import genai
from google.genai import types
import sacrebleu

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
PROMPTS = "~/b2s-eval/prompts"
OUT = f"~/b2s-eval/outputs/{MODEL}"
WORKERS = int(os.environ.get("WORKERS", "8"))
client = genai.Client(vertexai=True, api_key=open("~/.gemini_key").read().strip())

def call(prompt):
    for attempt in range(6):
        try:
            r = client.models.generate_content(model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096))
            return r.text or "", getattr(r.usage_metadata, "total_token_count", None)
        except Exception as e:
            print("ERR", attempt, str(e)[:150], file=sys.stderr, flush=True)
            if attempt == 5: return f"__ERROR__ {e}", None
            time.sleep(2 ** attempt)

def clean(t):
    t = t.strip()
    if "\n" in t and len(t.splitlines()) > 1:
        t = [l for l in t.splitlines() if l.strip()][-1]
    return t.strip().strip('"')

def run_lang(path):
    lang = os.path.basename(path)[:-6]
    os.makedirs(OUT, exist_ok=True)
    outp = f"{OUT}/{lang}.jsonl"
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            r = json.loads(l); done.add((r["direction"], r["idx"], r["context"]))
    rows = [json.loads(l) for l in open(path)]
    todo = [r for r in rows if (r["direction"], r["idx"], r["context"]) not in done]
    with open(outp, "a") as f, cf.ThreadPoolExecutor(WORKERS) as ex:
        for r, (txt, toks) in zip(todo, ex.map(lambda r: call(r["prompt"]), todo)):
            hyp = clean(txt)
            f.write(json.dumps({**{k: r[k] for k in r if k != "prompt"}, "raw": txt, "hyp": hyp, "tokens": toks,
                                "chrf": sacrebleu.sentence_chrf(hyp, [r["reference"]]).score}, ensure_ascii=False) + "\n"); f.flush()
    print(lang, "done", len(todo), "new", flush=True)

def summarize():
    import collections
    agg = collections.defaultdict(list)
    for p in glob.glob(f"{OUT}/*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if not r["raw"].startswith("__ERROR__"): agg[(r["lang"], r["direction"], r["context"])].append((r["hyp"], r["reference"]))
    print("lang\tdir\tcontext\tn\tchrF")
    for k in sorted(agg):
        h, ref = zip(*agg[k]); print(*k, len(h), f"{sacrebleu.corpus_chrf(list(h), [list(ref)]).score:.1f}", sep="\t")

if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]: summarize(); sys.exit()
    paths = [f"{PROMPTS}/{l}.jsonl" for l in sys.argv[1:]] or sorted(glob.glob(f"{PROMPTS}/*.jsonl"))
    for p in paths: run_lang(p)
    summarize()
