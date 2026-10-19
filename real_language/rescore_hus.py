"""Re-extract hypotheses from raw Hus outputs with a tag-aware cleaner and recompute chrF.
Fixes the eval_b2s.clean() first-line rule, which scored outputs of the form
"<translation>\n<text>" (no closing tag) as the literal "<translation>"."""
import json, glob, re, sys, collections
import sacrebleu

TR = {"mizo", "ilokano", "kimbundu"}

def clean(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    m = re.search(r"<translation>\s*(.*?)\s*(?:</translation>|$)", t, flags=re.S)
    if m: t = m.group(1)
    t = re.sub(r"^\s*(English|[A-Z][a-z]+)?\s*translation:\s*", "", t.strip(), flags=re.I)
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    t = paras[0] if paras else ""
    lines = [l for l in t.splitlines() if l.strip() and not re.fullmatch(r"</?translation>", l.strip())]
    t = lines[0] if lines else ""
    return t.strip().strip('"')

def main(root, tags):
    print("model\tgroup\tdir\tctx\tn\tchrF_old\tchrF_new\tchanged")
    for tag in tags:
        by = collections.defaultdict(lambda: ([], [], [], 0))
        for f in sorted(glob.glob(f"{root}/{tag}/*.jsonl")):
            for l in open(f):
                r = json.loads(l)
                if r["raw"].startswith("__ERROR__"): continue
                h = clean(r["raw"])
                k = ("trained" if r["lang"] in TR else "unseen", r["direction"], r["context"])
                o, n, ref, c = by[k]
                o.append(r["hyp"]); n.append(h); ref.append(r["reference"])
                by[k] = (o, n, ref, c + (h != r["hyp"]))
        for k in sorted(by):
            o, n, ref, c = by[k]
            print(f"{tag}\t{k[0]}\t{k[1]}\t{k[2]}\t{len(o)}\t{sacrebleu.corpus_chrf(o,[ref]).score:.1f}\t{sacrebleu.corpus_chrf(n,[ref]).score:.1f}\t{c}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
