import json,sys,collections
from score_cf import order
cf_file,test_file,out_file,tag=sys.argv[1:]
cf={json.loads(l)['instance_id']:json.loads(l) for l in open(cf_file)}
test=[json.loads(l) for l in open(test_file)]; test=[t for t in test if (t.get('ciphered_reference') or '').strip().strip('@').strip()]
out=[json.loads(l) for l in open(out_file)]; assert len(test)==len(out),(len(test),len(out))
agg=collections.defaultdict(collections.Counter)
for t,o in zip(test,out):
    r=cf.get(t['instance_id']); 
    if not r: continue
    part=t['instance_type'].split('.')[0]; b=order(o['pred'],r['orig_reference'],r['ciphered_reference'])
    agg[part][b]+=1; agg['all'][b]+=1
print(tag,'(ORIGINAL prompts)')
for k in sorted(agg): c=agg[k]; n=sum(c.values()); print(f"  {k:4s} n={n:3d} edited-order={c['edited']/n:5.1%} original-order={c['original']/n:5.1%} neither={c['neither']/n:5.1%}")
