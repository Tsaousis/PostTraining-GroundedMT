"""Language-level bootstrap CIs for the Back-to-School (Hus) evaluation.
Per language x direction x context: tag-aware corpus chrF (rescore_hus.clean).
Two effects on the 15 unseen languages (and the 3 trained):
  grammar  = book_dict - dict        (same dictionary entries, only the two book passages differ)
  transfer = arm(book_dict) - control(book_dict)
CI = percentile bootstrap over languages (resample languages with replacement, 2000 draws);
also a pooled sentence-level paired bootstrap for reference. Writes results/real_language/HUS_BOOTSTRAP_CI.tsv."""
import json, glob, sys, os, collections, random
import numpy as np, sacrebleu
from sacrebleu.metrics import CHRF as _C
_CHRF = _C()
sys.path.insert(0, os.path.dirname(__file__) + "/../real_language")
from rescore_hus import clean, TR

B, SEED = 2000, 0
ARMS = {  # tag: (dir, control_tag)
    "sft4b": ("./b2s-eval/outputs", "qwen3-4b-base"),
    "grpo4b-base-full": ("./swells-thesis/results/real_language/raw", "qwen3-4b-base"),
    "grpo4b-sft": ("./b2s-eval/outputs", "qwen3-4b-base"),
    "grpo4b-forced-s1": ("./b2s-eval/outputs", "qwen3-4b-base"),
    "grpo4b-forced-full": ("./b2s-eval/outputs", "qwen3-4b-base"),
    "grpo4b-base-ep2": ("./swells-thesis/results/real_language/raw", "qwen3-4b-base"),
    "grpo4b-base-sub2k": ("./swells-thesis/results/real_language/raw", "qwen3-4b-base"),
    "qwen3-4b-base": ("./b2s-eval/outputs", None),
    "grpo8b-cot-final": ("./swells-thesis/results/real_language/hus_outputs_swellsfmt", "qwen3-8b"),
    "qwen3-8b": ("./swells-thesis/results/real_language/hus_outputs_swellsfmt", None),
}

def load(tag):
    """-> {(lang,dir,ctx): (hyps, refs)} keyed by item idx for pairing"""
    d = collections.defaultdict(dict)
    for f in glob.glob(f"{ARMS[tag][0]}/{tag}/*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["raw"].startswith("__ERROR__"): continue
            d[(r["lang"], r["direction"], r["context"])][r.get("idx", r.get("original_id"))] = (clean(r["raw"]), r["reference"])
    return d

def chrf(pairs): return sacrebleu.corpus_chrf([h for h, _ in pairs], [[r for _, r in pairs]]).score

def lang_scores(d, ctx, direction):
    return {lang: chrf(list(v.values())) for (lang, dr, c), v in d.items() if dr == direction and c == ctx}

def ci_langs(diff):  # diff: {lang: delta}
    rng = random.Random(SEED); v = list(diff.values()); n = len(v)
    bs = [np.mean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B)]
    return np.mean(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def ci_sent(a, b, direction, ctxa, ctxb, langs):
    """pooled paired sentence bootstrap: resample items, corpus chrF difference"""
    pa, pb = [], []
    for (lang, dr, c), v in a.items():
        if dr != direction or c != ctxa or lang not in langs: continue
        w = b.get((lang, dr, ctxb), {})
        for i, hr in v.items():
            if i in w: pa.append(hr); pb.append(w[i])
    n = len(pa); rng = np.random.default_rng(SEED)
    sa = np.array(_CHRF._extract_corpus_statistics([h for h, _ in pa], [[r for _, r in pa]]), dtype=float)
    sb = np.array(_CHRF._extract_corpus_statistics([h for h, _ in pb], [[r for _, r in pb]]), dtype=float)
    f = lambda st: _CHRF._compute_f_score(list(st))
    idx = rng.integers(0, n, size=(500, n)); bs = [f(sa[i].sum(0)) - f(sb[i].sum(0)) for i in idx]
    return chrf(pa) - chrf(pb), np.percentile(bs, 2.5), np.percentile(bs, 97.5), n

rows = [("arm", "effect", "group", "dir", "n_langs", "mean", "ci_lo", "ci_hi", "pooled_delta", "pooled_lo", "pooled_hi", "n_items")]
data = {t: load(t) for t in ARMS}
for tag, (_, ctrl) in ARMS.items():
    d = data[tag]
    for group, langs in (("unseen", lambda l: l not in TR), ("trained", lambda l: l in TR)):
        for direction in ("ek", "ke"):
            bd, dc = lang_scores(d, "book_dict", direction), lang_scores(d, "dict", direction)
            L = sorted(l for l in bd if l in dc and langs(l))
            if L:
                m, lo, hi = ci_langs({l: bd[l] - dc[l] for l in L}); pd_, plo, phi, n = ci_sent(d, d, direction, "book_dict", "dict", set(L))
                rows.append((tag, "grammar(book_dict-dict)", group, direction, len(L), f"{m:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", f"{pd_:+.2f}", f"{plo:+.2f}", f"{phi:+.2f}", n))
            nn = lang_scores(d, "none", direction)
            L2 = sorted(l for l in dc if l in nn and langs(l))
            if L2:
                m, lo, hi = ci_langs({l: dc[l] - nn[l] for l in L2}); pd_, plo, phi, n = ci_sent(d, d, direction, "dict", "none", set(L2))
                rows.append((tag, "dictionary(dict-none)", group, direction, len(L2), f"{m:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", f"{pd_:+.2f}", f"{plo:+.2f}", f"{phi:+.2f}", n))
            if ctrl:
                c = lang_scores(data[ctrl], "book_dict", direction)
                L3 = sorted(l for l in bd if l in c and langs(l))
                m, lo, hi = ci_langs({l: bd[l] - c[l] for l in L3}); pd_, plo, phi, n = ci_sent(d, data[ctrl], direction, "book_dict", "book_dict", set(L3))
                rows.append((tag, f"transfer(vs {ctrl}, book_dict)", group, direction, len(L3), f"{m:+.2f}", f"{lo:+.2f}", f"{hi:+.2f}", f"{pd_:+.2f}", f"{plo:+.2f}", f"{phi:+.2f}", n))
out = "./swells-thesis/results/real_language/HUS_BOOTSTRAP_CI.tsv"
with open(out, "w") as f:
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
print(open(out).read())
