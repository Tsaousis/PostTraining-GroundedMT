import json,sys,collections,re,difflib
# usage: score_cf_wo_fuzzy.py cf_wo_cot_eng_art.jsonl outputs.jsonl TAG
# Locates S / V / O in the prediction by fuzzy token match (ratio>=0.6) and reports the 6-way constituent order.
cf=[json.loads(l) for l in open(sys.argv[1])]; out=[json.loads(l) for l in open(sys.argv[2])]; tag=sys.argv[3]
assert len(cf)==len(out)
def find(t,w,used):
    best=(0,None)
    for i,x in enumerate(t):
        if i in used: continue
        s=difflib.SequenceMatcher(None,x,w.lower()).ratio()
        if s>best[0]: best=(s,i)
    return best[1] if best[0]>=0.6 else None
def order(pred,r):
    t=re.findall(r"\S+",pred.lower()); used=set()
    idx={}
    for k,span in (('S',r['cf_subj']),('V',r['cf_verb']),('O',r['cf_obj'])):
        i=find(t,span[0],used)
        if i is None: return 'unparsed'
        used.add(i); idx[k]=i
    return ''.join(sorted(idx,key=idx.get))
c=collections.Counter(); per=collections.defaultdict(collections.Counter); lab=collections.Counter(); perl=collections.defaultdict(collections.Counter)
for r,o in zip(cf,out):
    assert o['gold']==r['ciphered_reference']
    od=order(o['pred'],r); c[od]+=1; per[r['cf_rule']][od]+=1
    edited={'SVO':'OVS','OVS':'SVO'}[r['cf_rule']]
    l='edited' if od==edited else 'original' if od==r['cf_rule'] else 'other'
    lab[l]+=1; perl[r['cf_rule']][l]+=1
n=len(cf)
print(tag,'n=%d'%n,' '.join('%s=%.1f%%'%(k,100*v/n) for k,v in sorted(lab.items())),'| orders:',' '.join('%s=%d'%(k,v) for k,v in c.most_common()))
for k in sorted(per):
    m=sum(per[k].values()); print('  orig %s -> edited %s: '%(k,{'SVO':'OVS','OVS':'SVO'}[k])+' '.join('%s=%.1f%%'%(l,100*x/m) for l,x in sorted(perl[k].items())),'| orders:',' '.join('%s=%d'%(a,b) for a,b in per[k].most_common()))
