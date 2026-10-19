import json,sys,collections,re
cf=[json.loads(l) for l in open(sys.argv[1])]
out=[json.loads(l) for l in open(sys.argv[2])]
tag=sys.argv[3]
assert len(cf)==len(out)
c=collections.Counter(); per=collections.defaultdict(collections.Counter)
for r,o in zip(cf,out):
    assert o['gold']==r['ciphered_reference']
    ci=r['orig_reference'].split()
    art=ci[r['cf_art_idx']].lower(); noun=ci[r['cf_noun_idx']].lower()
    toks=re.findall(r"\S+",o['pred'].lower())
    pairs=set(zip(toks,toks[1:]))
    new=tuple(r['ciphered_reference'].lower().split()[min(r['cf_art_idx'],r['cf_noun_idx']):][:2])
    old=tuple(ci[min(r['cf_art_idx'],r['cf_noun_idx']):][:2]); old=tuple(w.lower() for w in old)
    if new in pairs and old not in pairs: lab='edited'
    elif old in pairs and new not in pairs: lab='original'
    else: lab='neither'
    c[lab]+=1; per[r['cf_rule']][lab]+=1
n=len(cf)
print(tag,'n=%d'%n,' '.join('%s=%.1f%%'%(k,100*v/n) for k,v in sorted(c.items())))
for k,v in per.items():
    m=sum(v.values()); print('  orig rule "%s": '%k+' '.join('%s=%.1f%%'%(l,100*x/m) for l,x in sorted(v.items())))
