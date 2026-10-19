"""Evaluate a Gemini model (Vertex AI express API key) on the SWELLS test files.

Same scoring as evaluate.py (official normalisation/accuracy, chrF, per-partition
breakdown); per-row generations are cached so the run is resumable.
"""

import argparse
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import sacrebleu

from swells_reward import extract_translation

def norm(text):
    """Paper's normalisation: lowercased, stripped of full stops."""
    return (text or "").strip().strip("@").strip().lower().rstrip(".").strip()


def accuracy(pred, gold, adj_gold, partition):
    """Marmonier et al. accuracy: exact match 100, adjective-order-only error 50,
    bag-of-words matching for the Latin-derived partitions 8 and 9."""
    p, g = norm(pred), norm(gold)
    if not p:
        return 0.0
    if partition.startswith(("8", "9")):
        return 100.0 if sorted(p.split()) == sorted(g.split()) else 0.0
    if p == g:
        return 100.0
    if adj_gold and p == norm(adj_gold):
        return 50.0
    return 0.0


def load(path):
    direction = "eng_art" if "eng_art" in os.path.basename(path) else "art_eng"
    ciphered = direction == "eng_art"
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            gold = rec["ciphered_reference"] if ciphered else rec["reference"]
            gold = (gold or "").strip().strip("@").strip()
            adj_key = (
                "ciphered_adjective_order_error_reference"
                if ciphered
                else "adjective_order_error_reference"
            )
            if not gold:
                continue
            rows.append(
                {
                    "prompt": rec["prompt"],
                    "gold": gold,
                    "adj_gold": (rec.get(adj_key) or "").strip().strip("@").strip(),
                    "instance_type": rec.get("instance_type", ""),
                    "direction": direction,
                }
            )
    return rows


SYSTEM = "You are an expert linguist and translator."
URL = "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={key}"


def call(model, key, prompt, max_tokens, retries=8):
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
    }
    req = urllib.request.Request(
        URL.format(model=model, key=key),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=900))
            parts = d["candidates"][0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return text, d.get("usageMetadata", {}), d["candidates"][0].get("finishReason")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 503, 504) and attempt < retries - 1:
                time.sleep(min(120, 5 * 2**attempt))
                continue
            return "", {"error": f"{e.code} {msg}"}, "ERROR"
        except Exception as e:  # network / parse
            if attempt < retries - 1:
                time.sleep(min(120, 5 * 2**attempt))
                continue
            return "", {"error": repr(e)[:300]}, "ERROR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--key_file", default=os.path.expanduser("~/.gemini_key"))
    ap.add_argument("--test_files", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    key = open(args.key_file).read().strip()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for path in args.test_files:
        rows = load(path)
        name = os.path.basename(path).replace(".jsonl", "")
        cache_path = os.path.join(args.output_dir, f"{name}.raw.jsonl")
        cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                rec = json.loads(line)
                if rec.get("finish") != "ERROR":
                    cache[rec["i"]] = rec
        lock = threading.Lock()
        out_f = open(cache_path, "a")

        def work(i):
            if i in cache:
                return
            gen, usage, finish = call(args.model, key, rows[i]["prompt"], args.max_tokens)
            rec = {"i": i, "gen": gen, "usage": usage, "finish": finish}
            with lock:
                cache[i] = rec
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                done = len(cache)
                if done % 50 == 0:
                    print(f"{name}: {done}/{len(rows)}", flush=True)

        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, range(len(rows))))
        out_f.close()

        per_partition = defaultdict(list)
        per_type = defaultdict(list)
        scores, accs, records = [], [], []
        for i, r in enumerate(rows):
            gen = cache.get(i, {}).get("gen", "")
            pred = extract_translation(gen)
            s = sacrebleu.sentence_chrf(pred, [r["gold"]]).score if pred else 0.0
            partition = r["instance_type"].split(".")[0]
            if partition in ("6m", "6f"):
                partition = "6"
            a = accuracy(pred, r["gold"], r["adj_gold"], partition)
            scores.append(s)
            accs.append(a)
            per_type[r["instance_type"].split("__")[0]].append((s, a))
            per_partition[partition].append((s, a))
            records.append({"pred": pred, "gold": r["gold"], "chrf": s, "accuracy": a, "gen": gen})
        with open(os.path.join(args.output_dir, f"{name}.jsonl"), "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        errors = sum(1 for c in cache.values() if c.get("finish") == "ERROR") + (len(rows) - len(cache))
        summary[name] = {
            "model": args.model,
            "n": len(scores),
            "errors": errors,
            "chrf": statistics.mean(scores) if scores else 0.0,
            "exact_match": sum(1 for rec in records if rec["pred"].strip() == rec["gold"].strip()) / max(len(records), 1),
            "accuracy": statistics.mean(accs) if accs else 0.0,
            "by_phenomenon": {k: {"n": len(v), "chrf": statistics.mean(s for s, _ in v), "accuracy": statistics.mean(a for _, a in v)} for k, v in sorted(per_type.items())},
            "by_partition": {k: {"n": len(v), "chrf": statistics.mean(s for s, _ in v), "accuracy": statistics.mean(a for _, a in v)} for k, v in sorted(per_partition.items())},
            "tokens": {
                "prompt": sum(c.get("usage", {}).get("promptTokenCount", 0) for c in cache.values()),
                "output": sum(c.get("usage", {}).get("candidatesTokenCount", 0) for c in cache.values()),
                "thoughts": sum(c.get("usage", {}).get("thoughtsTokenCount", 0) for c in cache.values()),
            },
        }
        print(f"{name}: chrF {summary[name]['chrf']:.2f} acc {summary[name]['accuracy']:.1f} EM {summary[name]['exact_match']:.1%} (n={len(scores)}, errors={errors})", flush=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
