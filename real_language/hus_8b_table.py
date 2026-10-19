"""Per-language chrF table for the 8B Hus runs (SWELLS-format prompts, thinking on): outputs_swellsfmt/<tag>/*.jsonl -> TSV."""
import json,glob,collections,sacrebleu,sys
TAGS=sys.argv[1:] or ["grpo8b-cot-final","qwen3-8b"]
TR={"mizo","ilokano","kimbundu"}
res={}
for t in TAGS:
  by=collections.defaultdict(lambda:([],[]))
  for f in glob.glob(f"~/b2s-eval/outputs_swellsfmt/{t}/*.jsonl"):
    for l in open(f):
      r=json.loads(l); h,ref=by[(r["lang"],r["direction"],r["context"])]; h.append(r["hyp"]); ref.append(r["reference"])
  res[t]={k:(len(v[0]),sacrebleu.corpus_chrf(v[0],[v[1]]).score) for k,v in by.items()}
langs=sorted({k[0] for t in res for k in res[t]})
cols=[(t,d,c) for t in TAGS for d in ("ek","ke") for c in ("book_dict","dict")]
print("lang\tgroup\tn\t"+"\t".join(f"{t}:{d}:{c}" for t,d,c in cols))
means=collections.defaultdict(lambda:collections.defaultdict(list))
for L in langs:
  n=res[TAGS[0]].get((L,"ek","book_dict"),(0,0))[0]
  grp="trained" if L in TR else "unseen"
  row=[]
  for t,d,c in cols:
    v=res[t].get((L,d,c)); row.append(f"{v[1]:.1f}" if v else "")
    if v: means[(t,d,c)][grp].append(v[1])
  print(f"{L}\t{grp}\t{n}\t"+"\t".join(row))
for grp in ("trained","unseen","all"):
  row=[]
  for k in cols:
    xs=means[k]["trained"]+means[k]["unseen"] if grp=="all" else means[k][grp]
    row.append(f"{sum(xs)/len(xs):.1f}" if xs else "")
  print(f"MEAN_{grp}\t{grp}\t-\t"+"\t".join(row))
