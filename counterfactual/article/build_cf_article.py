import json,re,sys,collections
RULE={'precedes':'comes after','comes after':'precedes'}
src,dst=sys.argv[1],sys.argv[2]; c=collections.Counter()
with open(dst,'w') as f:
  for l in open(src):
    r=json.loads(l); pt=re.match(r'[0-9]+[a-z]?',r['instance_type']).group(0)
    if pt not in('1','4','6f','6m'): continue
    p=r['prompt']; m=re.search(r'The definite article always (precedes|comes after) its noun\.',p)
    if not m: c['norule']+=1; continue
    kind=m.group(1); after=kind=='comes after'
    forms=set(re.findall(r'the article is (\S+?)[.,]? ',p))|set(re.findall(r'The article is (\S+) for both',p))
    ci=r['ciphered_reference'].split(); idx=[i for i,w in enumerate(ci) if w in forms]
    if len(idx)!=1: c['nart']+=1; continue
    i=idx[0]; j=i-1 if after else i+1
    if j<0 or j>=len(ci): c['edge']+=1; continue
    ci2=ci[:]; ci2[i],ci2[j]=ci2[j],ci2[i]
    # edit the Definite-article section: rule sentence + quoted two-word examples
    s=p.find('** Definite article'); e=p.find('\n-----',s); seg=p[s:e]
    seg2=seg.replace(m.group(0),'The definite article always %s its noun.'%RULE[kind],1)
    seg2,nq=re.subn(r'"(\S+) (\S+)"',lambda mm:'"%s %s"'%(mm.group(2),mm.group(1)),seg2)
    assert seg2!=seg and nq>=3,(nq,seg[:200])
    r['prompt']=p[:s]+seg2+p[e:]
    r['orig_reference']=r['ciphered_reference']; r['ciphered_reference']=' '.join(ci2)
    r['cf']='article_pos_flip'; r['cf_rule']=kind; r['cf_art_idx']=i; r['cf_noun_idx']=j
    f.write(json.dumps(r,ensure_ascii=False)+'\n'); c[(pt,kind)]+=1
print(dict(c))
