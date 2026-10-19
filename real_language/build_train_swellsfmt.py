"""Re-render Hus-format MIK training prompts in the SWELLS CoT prompt format (the format used for the 8B Hus evals,
prompts_swellsfmt): dictionary entries as 'lemma : equivalent' lines, grammar passages as a Grammar Book Excerpts block,
'You may explain your chain of thoughts ...' suffix. Targets keep the gold translation inside <translation> tags."""
import json, re, sys
LANG = {"mizo": "Mizo", "ilokano": "Ilokano", "kimbundu": "Kimbundu"}
COT = ("You may explain your chain of thoughts prior to producing the required translation. IMPORTANT: Do write your "
       "translation between tags in the following manner: <translation>your translation here</translation>.")
def convert(u, lang, direction):
    L = LANG[lang]; src_l, tgt_l = (L, "English") if direction == "ke" else ("English", L)
    m = re.match(r"You are an expert translator\. Translate the following sentence from \w+ to \w+\s*: (.*?)\n", u, re.S)
    src = m.group(1).strip()
    ents = []
    for lem, eq in re.findall(r"\n(?:\w+) word(?: or phrase)?: (.*?)\n(?:\w+) translation: (.*?)\n", u):
        if (lem, eq) not in ents: ents.append((lem, eq))
    books = re.findall(r"grammar book:\n---\n(.*?)\n---", u, re.S)
    if direction == "ke": dict_fmt = f"{L} lemma : English equivalent"
    else: dict_fmt = f"English lemma : {L} equivalent"
    p = (f"{L} is a low-resource language. You are to translate the following {src_l} segment into {tgt_l} with the help of a few dictionary entries and excerpts from a grammar book.\n\n"
         f"Here is the text segment you must translate:\n{src}\n\n"
         f"Here are a few dictionary entries that may be of use to you; note that each entry follows the format: {dict_fmt}.\n"
         + "\n".join(f"{a} : {b}" for a, b in ents) + "\n\n"
         f"And here are relevant excerpts from a grammar book: \n\nBeginning of {L} Grammar Book Excerpts\n\n"
         + "\n\n".join(b.strip() for b in books) + f"\n\nEnd of {L} Grammar Book Excerpts\n\n\n"
         f"A reminder that the {src_l} sentence you must translate into {tgt_l} is:\n{src}\n\n{COT}")
    return p, len(ents), len(books)
n = 0; stats = []
with open(sys.argv[2], "w") as o:
    for l in open(sys.argv[1]):
        r = json.loads(l); u = r["messages"][1]["content"]
        p, ne, nb = convert(u, r["lang"], r["direction"]); stats.append((ne, nb, len(p)))
        r["messages"][1]["content"] = p; o.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
import statistics
print(n, "entries mean", statistics.mean(s[0] for s in stats), "min", min(s[0] for s in stats), "books mean", statistics.mean(s[1] for s in stats), "chars mean/max", statistics.mean(s[2] for s in stats), max(s[2] for s in stats), "zero-book", sum(s[1]==0 for s in stats))
