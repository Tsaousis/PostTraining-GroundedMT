"""chrF + tag compliance for Hus test items rendered in SWELLS CoT format; compares models on the shared language set."""
import json, glob, sys, collections, sacrebleu
TRAINED = {"mizo", "ilokano", "kimbundu"}
def load(d):
    by = collections.defaultdict(lambda: ([], [], []))
    for f in glob.glob(f"{d}/*.jsonl"):
        for l in open(f):
            r = json.loads(l); h, ref, t = by[(r["lang"], r["direction"], r["context"])]
            h.append(r["hyp"]); ref.append(r["reference"]); t.append(bool(r.get("has_tag")))
    return by
models = sys.argv[1:]
data = {m: load(f"raw_swellsfmt/{m}") for m in models}
langs = set.intersection(*[{k[0] for k in d} for d in data.values()])
print(f"shared languages: {len(langs)}: {' '.join(sorted(langs))}")
print("model\tgroup\tdir\tn_langs\tbook+dict\tdict\ttag%")
for m, by in data.items():
    for grp, L in [("all", langs), ("trained", langs & TRAINED), ("unseen", langs - TRAINED - {"kalamang"})]:
        for dr in ("ek", "ke"):
            s, tags = {}, []
            for c in ("book_dict", "dict"):
                v = [sacrebleu.corpus_chrf(by[(l, dr, c)][0], [by[(l, dr, c)][1]]).score for l in sorted(L) if (l, dr, c) in by]
                s[c] = sum(v) / len(v) if v else float("nan")
                for l in L: tags += by[(l, dr, c)][2]
            print(f"{m}\t{grp}\t{dr}\t{len(L)}\t{s['book_dict']:.1f}\t{s['dict']:.1f}\t{100*sum(tags)/max(1,len(tags)):.1f}")
print("\nper-language book+dict (ek/ke):")
for l in sorted(langs):
    print(l, *[f"{m}={sacrebleu.corpus_chrf(data[m][(l,'ek','book_dict')][0],[data[m][(l,'ek','book_dict')][1]]).score:.1f}/{sacrebleu.corpus_chrf(data[m][(l,'ke','book_dict')][0],[data[m][(l,'ke','book_dict')][1]]).score:.1f}" for m in models])
