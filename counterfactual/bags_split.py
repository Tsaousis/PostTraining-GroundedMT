import json,re,sys,collections
from score_cf import order
def bags(r):
    m=re.search(r"These are (.*?); e\.g\.",r['prompt'],re.S); gl=set()
    for g in re.findall(r"\(([^)]*)\)",m.group(1)): gl.update(x.strip().lower() for x in g.split('/'))
    src=set(re.findall(r"[a-z]+",r['source'].lower()))
    return bool(gl&src)
if __name__!="__main__": pass
if __name__=="__main__":
    cf=[json.loads(l) for l in open('cf_cot_eng_art.jsonl')]
    def run(tag,out,orig=False):
        agg=collections.defaultdict(collections.Counter)
        for r,o in zip(cf,out):
            b=order(o['pred'],r['orig_reference'],r['ciphered_reference'])
            agg[('BAGS' if bags(r) else 'nonBAGS')][b]+=1
        print(tag)
        for k,c in sorted(agg.items()):
            n=sum(c.values()); print(f"  {k:8s} n={n:3d} edited={c['edited']/n:5.1%} original={c['original']/n:5.1%} neither={c['neither']/n:5.1%}")
    import glob
    for tag,f in [('base4b CF','out/cf_base4b/cf_cot_eng_art.jsonl'),('sft4b CF (partial)','sft4b_partial.jsonl')]:
        out=[json.loads(l) for l in open(f)]; run(tag,out)
    # original prompts: align by instance_id
    def run_orig(tag,f):
        test=[json.loads(l) for l in open('CoT_full_eng_art_test_grammar_excerpts.jsonl')]; test=[t for t in test if (t.get('ciphered_reference') or '').strip().strip('@').strip()]
        out=[json.loads(l) for l in open(f)]; assert len(test)==len(out)
        byid={t['instance_id']:o for t,o in zip(test,out)}
        outs=[byid[r['instance_id']] for r in cf]; run(tag,outs)
    run_orig('base4b ORIG',sys.argv[1]); run_orig('sft4b ORIG',sys.argv[2]); run_orig('grpo8b ORIG',sys.argv[3])
