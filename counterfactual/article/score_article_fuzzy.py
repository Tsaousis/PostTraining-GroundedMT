import json,sys,collections,re,difflib
def lab(pred,art,noun,art_first):
    toks=re.findall(r"\S+",pred.lower()); 
    idx=[i for i,t in enumerate(toks) if t==art]
    if len(idx)!=1: return 'no_article'
    i=idx[0]
    sim=lambda j: difflib.SequenceMatcher(None,toks[j],noun).ratio() if 0<=j<len(toks) else 0
    l,r=sim(i-1),sim(i+1)
    if max(l,r)<0.5: return 'no_noun'
    noun_right = r>l
    return 'art_first' if noun_right else 'noun_first'
def run(cf,out,key,title):
    c=collections.Counter(); per=collections.defaultdict(collections.Counter)
    for r,o in zip(cf,out):
        ci=r['orig_reference'].lower().split(); art=ci[r['cf_art_idx']]; noun=ci[r['cf_noun_idx']]
        orig_art_first = r['cf_art_idx']<r['cf_noun_idx']
        L=lab(o['pred'],art,noun,orig_art_first)
        if L in('art_first','noun_first'):
            L = 'original' if (L=='art_first')==orig_art_first else 'edited'
        c[L]+=1; per[r['cf_rule']][L]+=1
    n=len(cf); print(title,'n=%d'%n,' '.join('%s=%.0f%%'%(k,100*v/n) for k,v in sorted(c.items())))
    for k,v in per.items():
        m=sum(v.values()); print('   rule "%s": '%k+' '.join('%s=%.0f%%'%(l,100*x/m) for l,x in sorted(v.items())))
cf=[json.loads(l) for l in open('cf_article_cot_eng_art.jsonl')]
src=[json.loads(l) for l in open('CoT_full_eng_art_test_grammar_excerpts.jsonl')]
for tag,orig in [('instruct8b','instruct8b_think_wg'),('sft8b_swells','sft8b_think_wg'),('grpo8b_cot','grpo8b_ep1_wg_think16k')]:
    oo=[json.loads(l) for l in open('results/raw/%s/CoT_full_eng_art_test_grammar_excerpts.jsonl'%orig)]
    bg={(s['prompt'][:400],s['ciphered_reference']):o for s,o in zip(src,oo)}
    run(cf,[bg[(r['prompt'][:400],r['orig_reference'])] for r in cf],None,tag+' ORIGINAL prompts ("original"=rule order)')
    try:
        out=[json.loads(l) for l in open('out/ar_%s/cf_article_cot_eng_art.jsonl'%tag)]
        assert len(out)==300 and all(o['gold']==r['ciphered_reference'] for r,o in zip(cf,out))
        run(cf,out,None,tag+' FLIPPED prompts')
    except FileNotFoundError: print(tag,'FLIPPED: not yet')
