"""Paired bootstrap (1000 resamples, corpus chrF) of grpo8b-cot-final vs qwen3-8b on Hus, by direction/context/group."""
import json,glob,collections,random,sacrebleu
TR={"mizo","ilokano","kimbundu"}
def load(t):
  d={}
  for f in glob.glob(f"~/b2s-eval/outputs_swellsfmt/{t}/*.jsonl"):
    for l in open(f):
      r=json.loads(l); d[(r["lang"],r["direction"],r["context"],r["idx"])]=(r["hyp"],r["reference"])
  return d
A,B=load("grpo8b-cot-final"),load("qwen3-8b"); keys=sorted(set(A)&set(B))
random.seed(0)
def chrf(h,r): return sacrebleu.corpus_chrf(h,[r]).score
for d in ("ek","ke"):
  for c in ("book_dict","dict"):
    for g in ("trained","unseen"):
      ks=[k for k in keys if k[1]==d and k[2]==c and ((k[0] in TR)==(g=="trained"))]
      ha=[A[k][0] for k in ks]; hb=[B[k][0] for k in ks]; rf=[A[k][1] for k in ks]
      obs=chrf(ha,rf)-chrf(hb,rf); wins=0; diffs=[]
      for _ in range(1000):
        idx=[random.randrange(len(ks)) for _ in ks]
        dd=chrf([ha[i] for i in idx],[rf[i] for i in idx])-chrf([hb[i] for i in idx],[rf[i] for i in idx]); diffs.append(dd); wins+=dd>0
      diffs.sort(); print(f"{d} {c} {g} n={len(ks)} grpo-base={obs:+.2f} CI95=[{diffs[25]:+.2f},{diffs[974]:+.2f}] p={1-wins/1000:.3f}")
