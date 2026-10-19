"""MIK train set with 5 LCS-retrieved parallel examples (pool = same language's train split, the item itself excluded).
Same rows/ids as b2s_train_mizo_ilokano_kimbundu.jsonl; only the user prompt differs."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_prompts import ROOT, split_book, build, load_pool
SYSTEM = "You are an expert linguist and translator."
SUFFIX = ("\n\nDo not explain your chain of thoughts, instead immediately produce the required translation in a "
          "spontaneous fashion. IMPORTANT: Do write your translation between tags in the following manner: "
          "<translation>your translation here</translation>.")
LANGS = ["mizo", "ilokano", "kimbundu"]
OUT = "./b2s-eval/train"; os.makedirs(OUT, exist_ok=True)
data = []
for lang in LANGS:
    res, spl = f"{ROOT}/resources/{lang}", f"{ROOT}/splits/{lang}"
    book = f"{res}/grammar_book_for_claude_long.txt"
    if not os.path.exists(book): book = f"{res}/grammar_book.txt"
    chunks = split_book(book)
    wl = json.load(open(f"{res}/wordlist.json"))
    word_ex = {"ke": [{lang: k, "english": v[1] if isinstance(v, list) else v} for k, v in wl["ke"].items()],
               "ek": [{lang: v if isinstance(v, str) else v[1], "english": k} for k, v in wl["ek"].items()]}
    for d in ("ke", "ek"):
        pool = load_pool(spl, d)
        items = [x for x in json.load(open(f"{spl}/train_examples_{d}.json")) if "original" in x]
        for i, it in enumerate(items):
            data.append({"lang": lang, "direction": d, "idx": i, "original_id": it.get("original_id"),
                         "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": build(lang, d, it["original"], word_ex[d], chunks, "book_dict_ex", pool, exclude_id=it.get("original_id")) + SUFFIX},
                                      {"role": "assistant", "content": f"<translation>{it['translation']}</translation>"}]})
        print(lang, d, len(items), flush=True)
random.Random(0).shuffle(data)
with open(f"{OUT}/b2s_train_mik_ex.jsonl", "w") as f:
    for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("TOTAL", len(data))
