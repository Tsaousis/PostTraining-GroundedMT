import json,sys,collections,re
cf=[json.loads(l) for l in open(sys.argv[1])]; out=[json.loads(l) for l in open(sys.argv[2])]; tag=sys.argv[3]
assert len(cf)==len(out)
c=collections.Counter(); per=collections.defaultdict(collections.Counter)
for r,o in zip(cf,out):
    assert o['gold']==r['ciphered_reference']
    i=r['cf_noun_idx']; new=r['ciphered_reference'].split()[i]; old=r['orig_reference'].split()[i]
    stem=new[len(r['cf_alt']):] if r['cf_prefix'] else new[:-len(r['cf_alt'])]
    toks=re.findall(r"\S+",o['pred'].lower())
    if new.lower() in toks: lab='edited'
    elif old.lower() in toks: lab='original'
    elif any(stem.lower() in t for t in toks): lab='stem_other'
    else: lab='neither'
    pt=re.match(r'[0-9]+[a-z]?',r['instance_type']).group(0); per[pt][lab]+=1; c[lab]+=1
n=len(cf); print(tag,'n=%d'%n,' '.join('%s=%.1f%%'%(k,100*v/n) for k,v in sorted(c.items())))
for pt in sorted(per): m=sum(per[pt].values()); print(' ',pt,'n=%d'%m,' '.join('%s=%.0f%%'%(k,100*v/m) for k,v in sorted(per[pt].items())))
