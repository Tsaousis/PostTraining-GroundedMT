import json,sys,collections,re
# usage: score_cf_wo.py cf_wo_cot_eng_art.jsonl outputs.jsonl TAG [-v]
cf=[json.loads(l) for l in open(sys.argv[1])]; out=[json.loads(l) for l in open(sys.argv[2])]; tag=sys.argv[3]
assert len(cf)==len(out)
def pos(t,span):
    span=[w.lower() for w in span]
    for i in range(len(t)-len(span)+1):
        if t[i:i+len(span)]==span: return i
def lab(pred,r):
    t=re.findall(r"\S+",pred.lower()); ps,pv,po=pos(t,r['cf_subj']),pos(t,r['cf_verb']),pos(t,r['cf_obj'])
    if None in (ps,pv,po): return 'neither'
    svo=ps<pv<po; ovs=po<pv<ps
    if r['cf_rule']=='SVO': return 'edited' if ovs else 'original' if svo else 'neither'
    return 'edited' if svo else 'original' if ovs else 'neither'
c=collections.Counter(); per=collections.defaultdict(collections.Counter); ex=collections.defaultdict(list)
for r,o in zip(cf,out):
    assert o['gold']==r['ciphered_reference']
    l=lab(o['pred'],r); c[l]+=1; per[r['cf_rule']][l]+=1; ex[(r['cf_rule'],l)].append((o['pred'][:80],r['orig_reference'],r['ciphered_reference']))
n=len(cf); print(tag,'n=%d'%n,' '.join('%s=%.1f%%'%(k,100*v/n) for k,v in sorted(c.items())))
for k,v in sorted(per.items()):
    m=sum(v.values()); print('  orig rule %s: '%k+' '.join('%s=%.1f%%'%(l,100*x/m) for l,x in sorted(v.items())))
if '-v' in sys.argv:
    for k,v in sorted(ex.items()):
        print('==',k); [print('  PRED:',a,'| ORIG:',b,'| EDIT:',d) for a,b,d in v[:3]]
