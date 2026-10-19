import json,glob,os,sys,collections,sacrebleu
model=sys.argv[1] if len(sys.argv)>1 else "gemini-3.1-pro-preview"
by=collections.defaultdict(lambda:([],[]))
err=0
for f in sorted(glob.glob(f"outputs/{model}/*.jsonl")):
    for l in open(f):
        r=json.loads(l)
        if r["raw"].startswith("__ERROR__"): err+=1; continue
        h,ref=by[(r["lang"],r["direction"],r["context"])]; h.append(r["hyp"]); ref.append(r["reference"])
ctxs=["book_dict","dict","none"]
print("lang\tdir\tn\tbook+dict\tdict\tnone\tbook-dict\tdict-none")
rows=[]
for lang in sorted({k[0] for k in by}):
    for d in ("ke","ek"):
        s={c:sacrebleu.corpus_chrf(by[(lang,d,c)][0],[by[(lang,d,c)][1]]).score for c in ctxs if (lang,d,c) in by}
        if len(s)<3: continue
        n=len(by[(lang,d,"none")][0]); rows.append((d,s))
        print(f"{lang}\t{d}\t{n}\t{s['book_dict']:.1f}\t{s['dict']:.1f}\t{s['none']:.1f}\t{s['book_dict']-s['dict']:+.1f}\t{s['dict']-s['none']:+.1f}")
for d in ("ke","ek"):
    m={c:sum(s[c] for dd,s in rows if dd==d)/max(1,sum(dd==d for dd,_ in rows)) for c in ctxs}
    print(f"MEAN\t{d}\t-\t{m['book_dict']:.1f}\t{m['dict']:.1f}\t{m['none']:.1f}\t{m['book_dict']-m['dict']:+.1f}\t{m['dict']-m['none']:+.1f}")
print("errors:",err,file=sys.stderr)
