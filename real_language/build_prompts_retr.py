"""Grammar-retrieval ablation prompts (inference only). Dictionary entries and 5 LCS-retrieved parallel
examples are identical across contexts; only how the 2 grammar passages are chosen differs:
  dict_ex          no grammar
  lcs_ex           current char-LCS passage retrieval (= book_dict_ex)
  emb_ex           multilingual-e5 embedding retrieval (query = source sentence)
  rerank_ex        e5 top-10 candidates re-ranked by Gemini 2.5 Flash for usefulness to this translation
Writes $B2S_OUT/<lang>.jsonl (default ~/b2s-eval/prompts_retr) and a retrieval log per language."""
import json, os, sys, re, random, concurrent.futures as cf
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_prompts as bp
from sentence_transformers import SentenceTransformer

OUT = os.environ.get("B2S_OUT", "~/b2s-eval/prompts_retr")
TOPN, K = 10, bp.K_PASS
emb_model = SentenceTransformer("intfloat/multilingual-e5-base")
_gem = None
def gemini():
    global _gem
    if _gem is None:
        from google import genai
        _gem = genai.Client(vertexai=True, api_key=open("~/.gemini_key").read().strip())
    return _gem

def rerank(lang, direction, source, cands):
    L = lang.capitalize()
    task = f"from {L} to English" if direction == "ke" else f"from English to {L}"
    lst = "\n\n".join(f"[{i}]\n{c[:1800]}" for i, c in enumerate(cands))
    p = (f"A translator must translate this sentence {task}:\n\n{source}\n\n"
         f"Below are {len(cands)} passages from a {L} grammar book. Pick the {K} passages whose grammatical "
         f"information (morphology, tense/aspect, word order, agreement, function words) is most useful for producing "
         f"this translation. Do NOT translate. Answer with only the {K} indices, comma-separated.\n\n{lst}")
    for _ in range(3):
        try:
            t = gemini().models.generate_content(model="gemini-2.5-flash", contents=p).text
            ids = [int(x) for x in re.findall(r"\d+", t) if int(x) < len(cands)]
            ids = list(dict.fromkeys(ids))[:K]
            if len(ids) == K: return ids, t
        except Exception as e: err = str(e)[:200]
    return list(range(K)), "FALLBACK"

def with_grammar(prompt_dict_ex, lang, passages):
    """insert grammar passages before the first retrieved parallel example (same layout as build_prompts.build)"""
    L = lang.capitalize()
    blocks = [f"To help with the translation, here is a passage retrieved from a {L}-English grammar book:\n---\n{p}\n---" for p in passages]
    marker = "To help with the translation, here is a similar"
    i = prompt_dict_ex.index(marker)
    return prompt_dict_ex[:i] + "\n".join(blocks) + "\n" + prompt_dict_ex[i:]

def main(langs):
    os.makedirs(OUT, exist_ok=True)
    for lang in langs:
        if os.path.exists(f"{OUT}/{lang}.jsonl"): continue
        res, spl = f"{bp.ROOT}/resources/{lang}", f"{bp.ROOT}/splits/{lang}"
        book = f"{res}/grammar_book_for_claude_long.txt"
        if not os.path.exists(book): book = f"{res}/grammar_book.txt"
        chunks = bp.split_book(book)
        cemb = emb_model.encode([f"passage: {c}" for c in chunks], normalize_embeddings=True, batch_size=32)
        wl = json.load(open(f"{res}/wordlist.json"))
        word_ex = {"ke": [{lang: k, "english": v[1] if isinstance(v, list) else v} for k, v in wl["ke"].items()],
                   "ek": [{lang: v if isinstance(v, str) else v[1], "english": k} for k, v in wl["ek"].items()]}
        rows, log = [], []
        for d in ("ke", "ek"):
            pool = bp.load_pool(spl, d)
            items = [x for x in json.load(open(f"{spl}/test_examples_{d}.json")) if "original" in x]
            rng = random.Random(bp.SEED); idx = sorted(rng.sample(range(len(items)), min(bp.N_PER_DIR, len(items))))
            sel = [items[i] for i in idx]
            base = [bp.build(lang, d, it["original"], word_ex[d], chunks, "dict_ex", pool, None, it["ground_truth"]) for it in sel]
            lcs = [bp.select_passages(it["original"], chunks, K) for it in sel]
            q = emb_model.encode([f"query: {it['original']}" for it in sel], normalize_embeddings=True)
            top = np.argsort(-(q @ cemb.T), axis=1)[:, :TOPN]
            emb = [[chunks[j] for j in t[:K]] for t in top]
            with cf.ThreadPoolExecutor(8) as ex:
                rr = list(ex.map(lambda a: rerank(lang, d, a[0]["original"], [chunks[j] for j in a[1]]), zip(sel, top)))
            for n, it in enumerate(sel):
                ids, raw = rr[n]
                variants = {"dict_ex": base[n], "lcs_ex": with_grammar(base[n], lang, lcs[n]),
                            "emb_ex": with_grammar(base[n], lang, emb[n]),
                            "rerank_ex": with_grammar(base[n], lang, [chunks[top[n][j]] for j in ids])}
                for ctx, pr in variants.items():
                    rows.append({"lang": lang, "direction": d, "idx": idx[n], "original_id": it.get("original_id"), "context": ctx,
                                 "source": it["original"], "reference": it["ground_truth"], "prompt": pr})
                log.append({"lang": lang, "direction": d, "idx": idx[n], "lcs": [chunks.index(c) for c in lcs[n]],
                            "emb_top10": top[n].tolist(), "rerank": [int(top[n][j]) for j in ids], "rerank_raw": raw})
        with open(f"{OUT}/{lang}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(f"{OUT}/{lang}.retrieval.json", "w") as f: json.dump(log, f)
        toks = max(len(bp.enc.encode(r["prompt"])) for r in rows)
        print(lang, len(rows), "rows, max tokens", toks, flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])
