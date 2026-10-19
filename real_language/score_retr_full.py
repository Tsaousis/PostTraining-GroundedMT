"""Aggregate the 15-language retrieval run: per-language chrF per grammar variant, pooled chrF,
language-level bootstrap CI (resample languages) and pooled paired sentence bootstrap of each variant vs dict_ex and rerank_ex vs lcs_ex.
usage: python score_retr_full.py <outputs_root> > RETRIEVAL_FULL.txt"""
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from rescore_hus import clean
from sacrebleu.metrics import CHRF
C = CHRF(); root = sys.argv[1]; CTX = ("dict_ex", "lcs_ex", "emb_ex", "rerank_ex"); rng = np.random.default_rng(0)
def stats(pairs): return np.array(C._extract_corpus_statistics([h for h, _ in pairs], [[r for _, r in pairs]]), dtype=float)
def f(st): return C._compute_f_score(list(st))
for mf in sorted(glob.glob(f"{root}/*")):
    tag = os.path.basename(mf); data = {}  # (lang, d, ctx) -> {idx: (hyp, ref)}
    for fn in sorted(glob.glob(f"{mf}/*.jsonl")):
        lang = os.path.basename(fn)[:-6]
        for l in open(fn):
            r = json.loads(l); data.setdefault((lang, r["direction"], r["context"]), {})[r["idx"]] = (clean(r["raw"]), r["reference"])
    langs = sorted({k[0] for k in data})
    for d in ("ek", "ke"):
        print(f"\n== {tag} {d} ==  (chrF; Δ vs dict_ex)")
        per = {c: {} for c in CTX}; st = {c: [] for c in CTX}
        for lang in langs:
            ids = sorted(set.intersection(*[set(data.get((lang, d, c), {})) for c in CTX]))
            if not ids: continue
            line = f"{lang:13s} n={len(ids):3d}"
            for c in CTX:
                s = stats([data[(lang, d, c)][i] for i in ids]); st[c].append(s); per[c][lang] = f(s.sum(0))
                line += f"  {c}={per[c][lang]:5.1f}" + (f"({per[c][lang]-per['dict_ex'][lang]:+.1f})" if c != "dict_ex" else "")
            print(line)
        pooled = {c: f(np.concatenate(st[c]).sum(0)) for c in CTX}
        print("POOLED        " + "  ".join(f"{c}={pooled[c]:5.1f}" for c in CTX))
        L = list(per["dict_ex"]); n = len(L)
        for a, b in (("lcs_ex", "dict_ex"), ("emb_ex", "dict_ex"), ("rerank_ex", "dict_ex"), ("rerank_ex", "lcs_ex"), ("emb_ex", "lcs_ex")):
            dl = np.array([per[a][l] - per[b][l] for l in L]); bl = [dl[rng.integers(0, n, n)].mean() for _ in range(2000)]
            sa, sb = np.concatenate(st[a]), np.concatenate(st[b]); m = len(sa)
            bs = [f(sa[i].sum(0)) - f(sb[i].sum(0)) for i in rng.integers(0, m, size=(500, m))]
            print(f"{a:9s}-{b:8s}: lang-level mean {dl.mean():+.2f} [{np.percentile(bl,2.5):+.2f},{np.percentile(bl,97.5):+.2f}] "
                  f"(langs>0: {(dl>0).sum()}/{n})  pooled {pooled[a]-pooled[b]:+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
