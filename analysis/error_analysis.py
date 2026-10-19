"""Qualitative error analysis: SWELLS (art->eng, W+G) and Hus real languages.
Prints per-model error-type counts and sampled positive/negative traces."""
import json, re, random, collections, glob, sys, os
random.seed(7)
TEST='~/swells_min/SWELLS_Dataset/Test/Test_CoT/CoT_full_art_eng_test_grammar_excerpts.jsonl'
T=[json.loads(l) for l in open(TEST)]
def part(t):
    p=t['instance_type'].split('.')[0]; return '6' if p.startswith('6') else p
ORDER=['1','2m','2f','3','4','5m','5f','6','7m','7f','8a','8g','9a','9g']
R='results'
SWELLS={
 '4B base (untrained, CoT prompt)':'~/results/base4b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B curriculum SFT':'~/results/sft_curriculum_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B GRPO-from-base ep3 (no-CoT)':f'{R}/raw/grpo_base_ep3/grpo_base_ep3_wg_nocot_4k/no_CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B base (untrained)':'~/results/base8b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B GRPO course7 (think off)':f'{R}/raw/grpo8b_course7_wg/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B SFT (think on)':f'{R}/raw/sft8b_think_wg/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B CoT-GRPO epoch 1 (think on)':f'{R}/raw/grpo8b_ep1_wg_think16k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B CoT-GRPO final 2 ep (think on)':'/tmp/g8b_ae.jsonl',
}
PRON={'i','you','he','she','it','we','they'}
def words(s): return re.findall(r"[a-z']+",s.lower())
def errtype(gold,pred):
    g,p=words(gold),words(pred)
    if not p: return 'empty/format'
    if set(g)==set(p) and g!=p: return 'word order'
    sg,sp=set(g),set(p)
    diff=(sg^sp)
    if not diff: return 'other'
    if diff<=PRON|{'his','her','their','my','your','our'}: return 'person/pronoun'
    # tense/aux: differences only in verb-ish tokens with shared stem
    aux={'will','would','was','were','is','are','am','had','has','have','been','being','did','do','does'}
    if diff<=aux: return 'tense/aux'
    stems=lambda S:{w[:4] for w in S if w not in aux}
    if stems(sg-sp)==stems(sp-sg) and stems(sg-sp): return 'tense/inflection'
    plural=lambda S:{w.rstrip('s') for w in S}
    if plural(sg)==plural(sp): return 'number'
    if len(sg-sp)+len(sp-sg)>=max(2,len(sg)//2): return 'lexical/garbled'
    return 'lexical (1-2 words)'
def tail(gen,n=450):
    th=gen.split('</think>')[0] if '</think>' in gen else gen.split('<translation>')[0]
    return re.sub(r'\s+',' ',th[-n:])
out=[]
P=print
P('# SWELLS art->eng W+G: per-model analysis\n')
for name,f in SWELLS.items():
    if not os.path.exists(f): P(f'## {name}: MISSING {f}\n'); continue
    M=[json.loads(l) for l in open(f)]
    if len(M)!=1400: P(f'## {name}: n={len(M)} skip'); continue
    assert all(m['gold']==t['target'] if 'target' in t else True for m,t in zip(M,T))
    acc=collections.defaultdict(list); et=collections.Counter(); etp=collections.defaultdict(collections.Counter)
    lens=[]; trunc=0; empty=0
    for t,m in zip(T,M):
        p=part(t); acc[p].append(m['accuracy']); lens.append(len(m['gen']))
        if '</translation>' not in m['gen']: trunc+=1
        if not m['pred'].strip(): empty+=1
        if m['accuracy']<100:
            e=errtype(m['gold'],m['pred']); et[e]+=1; etp['latin' if p in('8a','8g','9a','9g') else ('6-7' if p in('6','7m','7f') else '1-5')][e]+=1
    P(f'## {name}\n')
    P('acc by partition: '+' '.join(f'{p}:{sum(acc[p])/len(acc[p]):.0f}' for p in ORDER)+f' | avg {sum(sum(v) for v in acc.values())/1400:.1f}')
    P(f'n fail {sum(et.values())}, no closing tag {trunc}, empty {empty}, median gen chars {sorted(lens)[700]}')
    P('error types (all fails): '+', '.join(f'{k} {v}' for k,v in et.most_common()))
    for grp in ['1-5','6-7','latin']: P(f'  {grp}: '+', '.join(f'{k} {v}' for k,v in etp[grp].most_common(4)))
    P('\nNEGATIVE examples:')
    for pt in ['4','5m','6','7m','8a','9g']:
        bad=[(t,m) for t,m in zip(T,M) if part(t)==pt and m['accuracy']<50]
        if bad:
            t,m=random.choice(bad); P(f'- [{pt}] SRC `{t["ciphered_source"]}` GOLD "{m["gold"]}" PRED "{m["pred"]}" ({errtype(m["gold"],m["pred"])})\n  reasoning tail: …{tail(m["gen"])}')
    P('\nPOSITIVE examples:')
    for pt in ['3','5f','7f','8a','9a']:
        good=[(t,m) for t,m in zip(T,M) if part(t)==pt and m['accuracy']==100]
        if good:
            t,m=random.choice(good); P(f'- [{pt}] SRC `{t["ciphered_source"]}` GOLD "{m["gold"]}" PRED "{m["pred"]}"\n  reasoning tail: …{tail(m["gen"])}')
    P()
# ---------------- Hus
P('\n\n# Hus real languages (chrF, book+dict context)\n')
HUS={
 'Qwen3-4B base (untrained, Hus prompt)':'~/b2s-eval/outputs/qwen3-4b-base',
 'Qwen3-4B real-language SFT':'~/b2s-eval/outputs/sft4b',
 'Qwen3-4B GRPO-from-base (sub2k)':'~/b2s-eval/outputs/grpo4b-base',
 'Qwen3-4B GRPO-from-base (full 5,982)':f'{R}/real_language/raw/grpo4b-base-full',
 'Qwen3-4B GRPO-from-SFT':'~/b2s-eval/outputs/grpo4b-sft',
 'Llama-3.2-3B real-language SFT':f'{R}/real_language/hus_outputs_llama3b-sft',
 'Qwen3-4B base, SWELLS-format prompt':f'{R}/real_language/raw_swellsfmt/qwen3-4b-base',
 'SWELLS-trained 4B SFT, SWELLS-format prompt':f'{R}/real_language/raw_swellsfmt/swells-sft4b',
 'Gemini 3.1 Pro':'~/b2s-eval/outputs/gemini-3.1-pro-preview',
}
TR={'mizo','ilokano','kimbundu'}
for name,d in HUS.items():
    rows=[json.loads(l) for f in glob.glob(d+'/*.jsonl') for l in open(f)]
    rows=[r for r in rows if r['context']=='book_dict']
    if not rows: P(f'## {name}: no rows'); continue
    P(f'## {name}  (n={len(rows)})')
    by=collections.defaultdict(list)
    for r in rows: by[('trained' if r['lang'] in TR else 'unseen', r['direction'])].append(r)
    for k in sorted(by):
        v=by[k]; c=sum(r['chrf'] for r in v)/len(v)
        hl=sorted(len(r['hyp']) for r in v)[len(v)//2]; rl=sorted(len(r['reference']) for r in v)[len(v)//2]
        emp=sum(1 for r in v if not r['hyp'].strip()); long=sum(1 for r in v if len(r['hyp'])>3*max(1,len(r['reference'])))
        copy=sum(1 for r in v if r['hyp'].strip().lower()==r['source'].strip().lower())
        P(f'  {k[0]:7s} {k[1]}: chrF {c:.1f} | median hyp len {hl} vs ref {rl} | empty {emp} | >3x too long {long} | copied source {copy}')
    for key,lab in [(('unseen','ek'),'unseen en->X'),(('unseen','ke'),'unseen X->en'),(('trained','ke'),'trained X->en')]:
        v=by.get(key,[])
        if not v: continue
        v=sorted(v,key=lambda r:r['chrf'])
        P(f'  NEG {lab}:')
        for r in random.sample(v[:len(v)//4],2): P(f'   - {r["lang"]} chrF {r["chrf"]:.0f} | SRC "{r["source"][:90]}" | REF "{r["reference"][:90]}" | HYP "{r["hyp"][:140]}"')
        P(f'  POS {lab}:')
        for r in random.sample(v[-len(v)//4:],2): P(f'   - {r["lang"]} chrF {r["chrf"]:.0f} | SRC "{r["source"][:90]}" | REF "{r["reference"][:90]}" | HYP "{r["hyp"][:140]}"')
    P()
