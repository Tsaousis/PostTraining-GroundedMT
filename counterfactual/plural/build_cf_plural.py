import json,re,sys,collections
SUF=re.compile(r"The plural is usually formed from the singular by adding the (prefix )?morpheme (\S+?) \(e\.g\., (\S+) > (\S+) 'houses'\)\.\s*Nouns (ending in|beginning with) (\S+) and (\S+) take the (ending|prefix) (\S+) instead of (\S+)",re.S)
DET={'les','des','aux','ces','ses','mes','tes','nos','vos','leurs'}
src,dst=sys.argv[1],sys.argv[2]; c=collections.Counter()
with open(dst,'w') as f:
  for l in open(src):
    r=json.loads(l); pt=re.match(r'[0-9]+[a-z]?',r['instance_type']).group(0)
    if pt not in('4','5f','5m','6f','6m','7f','7m'): continue
    m=SUF.search(r['prompt'])
    if not m: c['nopat']+=1; continue
    pre,reg,sg,pl,_,k1,k2,_,alt,reg2=m.groups(); pre=bool(pre)
    reg=reg.strip('-'); alt=alt.strip('-'); assert reg2.strip('-')==reg
    fr=r['reference'].strip('@').split(); ci=r['ciphered_reference'].split()
    assert len(fr)==len(ci),(fr,ci)
    idx=[i for i,w in enumerate(fr) if i>0 and fr[i-1] in DET and w.endswith('s') and not w.endswith(('aus','eus')) and not w[:-1].endswith('s')]
    if len(idx)!=1: c['skip']+=1; continue
    i=idx[0]; w=ci[i]
    ok = w.startswith(reg) if pre else w.endswith(reg)
    if not ok: c['morph_mismatch']+=1; continue
    new = alt+w[len(reg):] if pre else w[:-len(reg)]+alt
    ci2=ci[:]; ci2[i]=new
    # edit rule: swap reg<->alt in the two sentences
    s,e=m.span(); seg=r['prompt'][s:e]
    seg2=seg.replace(f"morpheme {('' if pre else '-')}{reg}",f"morpheme {('' if pre else '-')}{alt}",1)
    seg2=re.sub(r"(take the (?:ending|prefix) )(\S+)( instead of )(\S+)",lambda mm: mm.group(1)+mm.group(4)+mm.group(3)+mm.group(2),seg2,1)
    seg2=seg2.replace(f"> {pl} 'houses'", f"> {(alt+sg) if pre else (sg+alt)} 'houses'",1)
    assert seg2!=seg
    r['prompt']=r['prompt'][:s]+seg2+r['prompt'][e:]
    r['orig_reference']=r['ciphered_reference']; r['ciphered_reference']=' '.join(ci2)
    r['cf']='plural_morph_swap'; r['cf_noun_idx']=i; r['cf_reg']=reg; r['cf_alt']=alt; r['cf_prefix']=pre
    f.write(json.dumps(r,ensure_ascii=False)+'\n'); c[pt]+=1
print(dict(c))
