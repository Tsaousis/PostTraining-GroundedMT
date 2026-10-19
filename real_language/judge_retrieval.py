"""Blind relevance judging of retrieved grammar passages (Gemini 2.5 Flash).
For each test item: the union of passages chosen by lcs / emb / rerank is shown once, shuffled, without method labels;
the judge scores each 0 (irrelevant), 1 (some relevant vocabulary/topic), 2 (contains a grammar rule needed for this sentence).
Output: per-method mean score and share of items with >=1 passage scored 2."""
import json, sys, re, random, collections, concurrent.futures as cf
sys.path.insert(0, "real_language")
import build_prompts as bp
from google import genai
cl = genai.Client(vertexai=True, api_key=open("~/.gemini_key").read().strip())
OUT = sys.argv[1]; langs = sys.argv[2:]

def judge(lang, d, src, ref, cands):
    L = lang.capitalize(); task = f"{L} -> English" if d == "ke" else f"English -> {L}"
    lst = "\n\n".join(f"[{i}]\n{c[:1800]}" for i, c in enumerate(cands))
    p = (f"Translation task ({task}).\nSource: {src}\nReference translation: {ref}\n\nRate each passage from a {L} grammar book for "
         f"how useful it is for producing this translation: 0 = irrelevant, 1 = relevant vocabulary or topic but no needed rule, "
         f"2 = states a grammatical rule or construction actually needed here (tense/aspect marker, agreement, word order, pronoun, case, negation...).\n"
         f"Answer as JSON list of {len(cands)} integers only.\n\n{lst}")
    for _ in range(3):
        try:
            t = cl.models.generate_content(model="gemini-2.5-flash", contents=p).text
            s = [int(x) for x in re.findall(r"\d", t)]; s = [min(x, 2) for x in s][:len(cands)]
            if len(s) == len(cands): return s
        except Exception: pass
    return None

for lang in langs:
    res = f"{bp.ROOT}/resources/{lang}"; book = f"{res}/grammar_book_for_claude_long.txt"
    import os
    if not os.path.exists(book): book = f"{res}/grammar_book.txt"
    chunks = bp.split_book(book)
    log = json.load(open(f"{OUT}/{lang}.retrieval.json"))
    rows = {(r["direction"], r["idx"]): r for r in map(json.loads, open(f"{OUT}/{lang}.jsonl")) if r["context"] == "dict_ex"}
    def one(l):
        r = rows[(l["direction"], l["idx"])]
        meth = {"lcs": l["lcs"], "emb": l["emb_top10"][:2], "rerank": l["rerank"]}
        ids = sorted(set(sum(meth.values(), []))); random.Random(l["idx"]).shuffle(ids)
        s = judge(lang, l["direction"], r["source"], r["reference"], [chunks[i] for i in ids])
        if s is None: return None
        sc = dict(zip(ids, s)); return {m: [sc[i] for i in v] for m, v in meth.items()}, l["direction"]
    with cf.ThreadPoolExecutor(8) as ex: res_ = [x for x in ex.map(one, log) if x]
    for d in ("ek", "ke"):
        print(f"{lang} {d}:", end=" ")
        for m in ("lcs", "emb", "rerank"):
            v = [x[0][m] for x in res_ if x[1] == d]
            print(f"{m} mean {sum(map(sum, v))/max(1,len(v))/2:.2f} any-rule {sum(max(a)==2 for a in v)/max(1,len(v)):.0%}", end=" | ")
        print(len([1 for x in res_ if x[1]==d]), "items")
    json.dump(res_, open(f"{OUT}/{lang}.judge.json", "w"))
