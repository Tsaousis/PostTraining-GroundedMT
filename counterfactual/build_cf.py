import json,re,sys
PAT=re.compile(r"Adjective placement follows specific patterns\. Most adjectives are placed (before|after) the noun they modify, and this is particularly true for colors, as in (\S+) (\S+) (\S+) \(the red wine\)\. However, certain short, commonly used adjectives – those dealing with beauty, age, goodness, or size \(sometimes remembered by the acronym \"BAGS\"\) – (must|typically) (precede|follow) their nouns\. These are (.*?); e\.g\.,? (\S+) (\S+) (\S+) \(\"a beautiful woman\"\)\.")
OPP={'before':'after','after':'before','precede':'follow','follow':'precede'}
def edit(p):
    m=PAT.search(p); assert m, p[-2500:]
    pos,a,w1,w2,mod,verb,lst,a2,x1,x2=m.groups()
    new=(f"Adjective placement follows specific patterns. Most adjectives are placed {OPP[pos]} the noun they modify, and this is particularly true for colors, as in {a} {w2} {w1} (the red wine). However, certain short, commonly used adjectives – those dealing with beauty, age, goodness, or size (sometimes remembered by the acronym \"BAGS\") – {mod} {OPP[verb]} their nouns. These are {lst}; e.g., {a2} {x2} {x1} (\"a beautiful woman\").")
    return p[:m.start()]+new+p[m.end():]
src,dst=sys.argv[1],sys.argv[2]; n=0
with open(dst,'w') as f:
    for l in open(src):
        r=json.loads(l)
        if not r.get('ciphered_adjective_order_error_reference'): continue
        r['prompt']=edit(r['prompt']); r['orig_reference']=r['ciphered_reference']
        r['ciphered_reference']=r['ciphered_adjective_order_error_reference']; r['cf']='adj_order_flip'
        f.write(json.dumps(r,ensure_ascii=False)+'\n'); n+=1
print(dst,n)
