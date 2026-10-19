import json,sys,re,collections,difflib
def norm(s): return re.sub(r"\s+"," ",s.strip().lower().strip(".!?"))
def toks(s): return re.findall(r"[^\s]+",norm(s))
def order(pred, orig, edit):
    """Return 'edited'/'original'/'neither' by relative order of the two swapped words."""
    ot,et=toks(orig),toks(edit)
    if len(ot)!=len(et) or sorted(ot)!=sorted(et): return 'neither'
    diff=[i for i,(a,b) in enumerate(zip(ot,et)) if a!=b]
    if len(diff)!=2: return 'neither'
    w1,w2=ot[diff[0]],ot[diff[1]]   # in original: w1 before w2 ; in edited: w2 before w1
    pt=toks(pred)
    def pos(w):
        best,bi=0,None
        for i,p in enumerate(pt):
            r=difflib.SequenceMatcher(None,p,w).ratio()
            if r>best: best,bi=r,i
        return bi if best>=0.6 else None
    p1,p2=pos(w1),pos(w2)
    if p1 is None or p2 is None or p1==p2: return 'neither'
    return 'original' if p1<p2 else 'edited'
def main(cf_file,out_file,tag):
    cf=[json.loads(l) for l in open(cf_file)]; out=[json.loads(l) for l in open(out_file)]
    assert len(cf)==len(out),(len(cf),len(out))
    agg=collections.defaultdict(collections.Counter); ex=collections.Counter()
    for r,o in zip(cf,out):
        part=r['instance_type'].split('.')[0]; b=order(o['pred'],r['orig_reference'],r['ciphered_reference'])
        agg[part][b]+=1; agg['all'][b]+=1
        p=norm(o['pred']); ex['edited_exact']+=p==norm(r['ciphered_reference']); ex['orig_exact']+=p==norm(r['orig_reference']); ex['notag']+= not o['pred'].strip()
    print(tag, dict(ex))
    for k in sorted(agg): c=agg[k]; n=sum(c.values()); print(f"  {k:4s} n={n:3d} edited={c['edited']/n:5.1%} original={c['original']/n:5.1%} neither={c['neither']/n:5.1%}")
if __name__=='__main__': main(*sys.argv[1:])
