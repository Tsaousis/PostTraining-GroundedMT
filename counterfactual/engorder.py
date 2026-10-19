import json,re,sys,collections
from score_cf import order
from bags_split import bags
cf=[json.loads(l) for l in open('cf_cot_eng_art.jsonl')]
def orig_adj_first(r):
    # cf prompt is flipped; read flipped values and invert
    m=re.search(r"Most adjectives are placed (before|after) the noun.*?\(sometimes remembered by the acronym \"BAGS\"\) – (?:must|typically) (precede|follow)",r['prompt'],re.S)
    most,bg=m.groups()
    if bags(r): return bg=='follow'      # flipped says follow -> orig precede
    return most=='after'                  # flipped says after -> orig before
def run(tag,out):
    agg=collections.defaultdict(collections.Counter)
    for r,o in zip(cf,out):
        b=order(o['pred'],r['orig_reference'],r['ciphered_reference'])
        oaf=orig_adj_first(r)
        key='origrule=adjfirst(Eng)' if oaf else 'origrule=nounfirst'
        if b=='neither': res='neither'
        else:
            adj_first = (b=='original')==oaf
            res='out=adjfirst(Eng)' if adj_first else 'out=nounfirst'
        agg[key][res]+=1
        if b!='neither': agg[key][b]+=1
    print(tag)
    for k in sorted(agg):
        c=agg[k]; n=c['out=adjfirst(Eng)']+c['out=nounfirst']+c['neither']
        print(f"  {k:24s} n={n:3d} follows-edit={c['edited']/n:4.0%} keeps-orig={c['original']/n:4.0%} | adj-first(Eng)={c['out=adjfirst(Eng)']/n:4.0%} noun-first={c['out=nounfirst']/n:4.0%}")
if __name__=='__main__':
    for tag,f in [('base4b CF','out/cf_base4b/cf_cot_eng_art.jsonl'),('sft4b CF','out/cf_sft4b/cf_cot_eng_art.jsonl')]:
        run(tag,[json.loads(l) for l in open(f)])
    test=[json.loads(l) for l in open('CoT_full_eng_art_test_grammar_excerpts.jsonl')]; test=[t for t in test if (t.get('ciphered_reference') or '').strip().strip('@').strip()]
    for tag,f in [('base4b ORIG',sys.argv[1]),('sft4b ORIG',sys.argv[2]),('grpo8b ORIG',sys.argv[3])]:
        out=[json.loads(l) for l in open(f)]; byid={t['instance_id']:o for t,o in zip(test,out)}
        run(tag,[byid[r['instance_id']] for r in cf])
