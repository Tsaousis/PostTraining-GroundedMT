import json,re,sys,collections
src,dst=sys.argv[1],sys.argv[2]
SVO=r'is subject-verb-object \(SVO\)\.\n\n\s+(\S+) (\S+) (\S+) \(I love chocolate\)\.'
OVS=r'is the rare object-verb-subject \(OVS\)\.\n\n\s+(\S+) (\S+) (\S+) \(I love chocolate\)\.'
c=collections.Counter()
with open(dst,'w') as f:
  for l in open(src):
    r=json.loads(l)
    if r['instance_type'][0] not in '45': continue
    fr=r['reference'].strip('@').split()
    if 'les' not in fr or "'" in fr[0]: c['elided']+=1; continue
    k=fr.index('les'); nobj=len(fr)-k
    ci=r['ciphered_reference'].split()
    if len(ci)!=len(fr): c['len']+=1; continue
    p=r['prompt']; s=p.find('WORD ORDER'); e=p.find('\n-----',s+20); seg=p[s:e]
    m=re.search(SVO,seg)
    if m:
        kind='SVO'; subj,verb,obj=ci[:1],ci[1:k],ci[k:]
        new=obj+verb+subj
        seg2=seg.replace(m.group(0),'is the rare object-verb-subject (OVS).\n\n    %s %s %s (I love chocolate).'%(m.group(2),m.group(3),m.group(1)))
    else:
        m=re.search(OVS,seg)
        if not m: c['nowo']+=1; continue
        kind='OVS'; obj,subj=ci[:nobj],ci[-1:]; verb=ci[nobj:-1]
        new=subj+verb+obj
        seg2=seg.replace(m.group(0),'is subject-verb-object (SVO).\n\n    %s %s %s (I love chocolate).'%(m.group(3),m.group(1),m.group(2)))
    assert len(verb)>=1 and seg2!=seg
    r['prompt']=p[:s]+seg2+p[e:]
    r['orig_reference']=r['ciphered_reference']; r['ciphered_reference']=' '.join(new)
    r['cf']='word_order_flip'; r['cf_rule']=kind; r['cf_subj']=subj; r['cf_verb']=verb; r['cf_obj']=obj
    f.write(json.dumps(r,ensure_ascii=False)+'\n'); c[(r['instance_type'][:2].rstrip('.'),kind)]+=1
print(dict(c))
