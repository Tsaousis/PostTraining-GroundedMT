"""SWELLS art->eng W+G error analysis -> LaTeX fragments + JSON for the paper appendix."""
import json,re,random,collections,os,sys
random.seed(11)
TEST='~/swells_min/SWELLS_Dataset/Test/Test_CoT/CoT_full_art_eng_test_grammar_excerpts.jsonl'
T=[json.loads(l) for l in open(TEST)]
def part(t):
    p=t['instance_type'].split('.')[0]; return '6' if p.startswith('6') else p
def grp(p): return 'Latin' if p in('8a','8g','9a','9g') else ('6--7' if p in('6','7m','7f') else '1--5')
R='results'
M={
 '4B base':'~/results/base4b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B SFT':'~/results/sft_curriculum_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B GRPO no-CoT':f'{R}/raw/grpo_base_ep3/grpo_base_ep3_wg_nocot_4k/no_CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B base':'~/results/base8b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B SFT':f'{R}/raw/sft8b_think_wg/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B CoT-GRPO ep1':f'{R}/raw/grpo8b_ep1_wg_think16k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B CoT-GRPO final':f'{R}/raw/grpo8b_cot_final_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
}
PRON={'i','you','he','she','it','we','they','his','her','their','my','your','our','him','them','us','me'}
AUX={'will','would','was','were','is','are','am','had','has','have','been','being','did','do','does'}
def words(s): return re.findall(r"[a-z']+",s.lower())
def errtype(gold,pred):
    g,p=words(gold),words(pred)
    if not p: return 'empty/format'
    sg,sp=set(g),set(p)
    if sg==sp and g!=p: return 'word order'
    diff=sg^sp
    if not diff: return 'other'
    if diff<=PRON: return 'person/pronoun'
    if diff<=AUX: return 'tense/aux'
    stems=lambda S:{w[:4] for w in S if w not in AUX}
    if stems(sg-sp)==stems(sp-sg) and stems(sg-sp): return 'tense/inflection'
    plural=lambda S:{w.rstrip('s') for w in S}
    if plural(sg)==plural(sp): return 'number'
    if len(sg-sp)+len(sp-sg)>=max(2,len(sg)//2): return 'lexical/garbled'
    return 'lexical (1--2 words)'
def tex(s): return s.replace('\\','').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#').replace('$','\\$').replace('{','\\{').replace('}','\\}')
def hl(gold,pred):
    """bold the words of pred that are not in gold"""
    sg=set(words(gold)); out=[]
    for w in pred.split():
        out.append(('\\textbf{'+tex(w)+'}') if re.sub(r"[^a-z']",'',w.lower()) not in sg else tex(w))
    return ' '.join(out)
def tail(gen,n=380):
    th=gen.split('</think>')[0] if '</think>' in gen else gen.split('<translation>')[0]
    return re.sub(r'\s+',' ',th[-n:]).strip()
# tense signature: which aux/ending verb form was produced for which gold form
def tense_sig(gold,pred):
    g,p=words(gold),words(pred)
    gv=[w for w in g if w in AUX]; pv=[w for w in p if w in AUX]
    return ' '.join(gv) or '(simple)', ' '.join(pv) or '(simple)'
data={}
for name,f in M.items():
    rows=[json.loads(l) for l in open(f)]; assert len(rows)==1400
    et=collections.defaultdict(collections.Counter); tense_conf=collections.Counter(); ex=collections.defaultdict(list)
    accg=collections.defaultdict(list)
    for t,m in zip(T,rows):
        p=part(t); g=grp(p); accg[g].append(m['accuracy'])
        if m['accuracy']<100:
            e=errtype(m['gold'],m['pred']); et[g][e]+=1
            if e in('tense/aux','tense/inflection'):
                tense_conf[tense_sig(m['gold'],m['pred'])]+=1
            ex[(g,e)].append((t,m))
    data[name]=dict(et=et,tense_conf=tense_conf,ex=ex,accg={g:sum(v)/len(v) for g,v in accg.items()},nfail=sum(sum(c.values()) for c in et.values()))
# ---- Table 1: error profile by group (LaTeX)
TYPES=['lexical/garbled','tense/inflection','tense/aux','lexical (1--2 words)','person/pronoun','number','word order']
SH=['garbled','tense','aux','1--2 w.','person','number','order']
L=[]
L.append(r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}')
L.append(r'\begin{tabular}{ll r rrrrrrr}\toprule')
L.append('Model & group & acc & '+' & '.join(SH)+r' \\ \midrule')
for name,d in data.items():
    for i,g in enumerate(['1--5','6--7','Latin']):
        c=d['et'][g]; L.append((name if i==0 else '')+f' & {g} & {d["accg"][g]:.0f} & '+' & '.join(str(c[t]) for t in TYPES)+r' \\')
    L.append(r'\addlinespace')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{SWELLS art\tr eng (W$+$G, 1,400 items): exact accuracy and counts of failures by automatically assigned error type, per partition group. ``tense'' = same stems, different inflection; ``aux'' = only auxiliaries differ (will/would/was/had\ldots); ``1--2 w.'' = one or two content words off; ``person'' = only pronouns differ; ``order'' = same words, different order.}\label{tab:err_profile}')
L.append(r'\end{table*}')
open('paper/gen/swells_err_profile.tex','w').write('\n'.join(L))
# ---- Table 2: tense confusion (gold aux -> pred aux) for SFT/GRPO models
L=[r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llr}\toprule','Model & gold form $\\to$ produced form & n \\\\ \\midrule']
for name in ['4B SFT','4B GRPO no-CoT','8B SFT','8B CoT-GRPO final']:
    for (g,p),n in data[name]['tense_conf'].most_common(5):
        L.append(f'{name} & {tex(g)} $\\to$ {tex(p)} & {n} \\\\')
    L.append(r'\addlinespace')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Most frequent tense/aspect confusions among tense-type failures (auxiliary pattern of the gold vs the prediction; ``(simple)'' $=$ no auxiliary, e.g. simple past/present).}\label{tab:tense_conf}\end{table}')
open('paper/gen/swells_tense_conf.tex','w').write('\n'.join(L))
# ---- Examples: tense errors with highlighted words, + reasoning tail for CoT models
def pick(name,g,e,k):
    xs=data[name]['ex'][(g,e)]; random.shuffle(xs); return xs[:k]
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{1.9cm}p{0.7cm}p{4.3cm}p{4.3cm}p{4.3cm}}\toprule','Model & part. & gold & prediction (wrong words in bold) & what the reasoning says \\\\ \\midrule']
for name in ['4B SFT','4B GRPO no-CoT','8B SFT','8B CoT-GRPO ep1','8B CoT-GRPO final']:
    for g in ['1--5','Latin']:
        for t,m in pick(name,g,'tense/aux',1)+pick(name,g,'tense/inflection',1):
            rs=tail(m['gen'],260) if 'think' in m['gen'] or 'no-CoT' not in name else '(direct answer, no reasoning)'
            L.append(f'{name} & {part(t)} & {tex(m["gold"])} & {hl(m["gold"],m["pred"])} & \\emph{{\\ldots{tex(rs)}}} \\\\ \\addlinespace')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Tense/aspect failures on SWELLS art\tr eng (randomly sampled from the tense-type failures of each model). Bold $=$ words not in the gold. The right column is the last 260 characters of the reasoning before the answer.}\label{tab:tense_examples}\end{table*}')
open('paper/gen/swells_tense_examples.tex','w').write('\n'.join(L))
# ---- Examples: other error classes (person, order, 1-2 words, garbled) per model
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{1.9cm}p{0.6cm}p{1.6cm}p{3.6cm}p{3.6cm}p{4.6cm}}\toprule','Model & part. & type & gold & prediction & reasoning tail \\\\ \\midrule']
for name in ['4B SFT','4B GRPO no-CoT','8B SFT','8B CoT-GRPO final']:
    for g,e in [('1--5','word order'),('6--7','person/pronoun'),('6--7','lexical (1--2 words)'),('Latin','lexical/garbled'),('Latin','number')]:
        for t,m in pick(name,g,e,1):
            rs=tail(m['gen'],220) if 'no-CoT' not in name else '(direct answer)'
            L.append(f'{name} & {part(t)} & {e} & {tex(m["gold"])} & {hl(m["gold"],m["pred"])} & \\emph{{\\ldots{tex(rs)}}} \\\\ \\addlinespace')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Other failure classes on SWELLS art\tr eng, one random example per model and class.}\label{tab:other_examples}\end{table*}')
open('paper/gen/swells_other_examples.tex','w').write('\n'.join(L))
# ---- Positive examples with full-ish reasoning for SFT vs GRPO on same Latin item
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{1.6cm}p{2.6cm}p{2.6cm}p{9.6cm}}\toprule','Model & gold & prediction & reasoning (tail) \\\\ \\midrule']
rows={n:[json.loads(l) for l in open(f)] for n,f in M.items()}
cands=[i for i,t in enumerate(T) if grp(part(t))=='Latin' and rows['8B CoT-GRPO final'][i]['accuracy']==100 and rows['8B SFT'][i]['accuracy']<100 and rows['4B SFT'][i]['accuracy']<100]
random.shuffle(cands)
for i in cands[:2]:
    t=T[i]; L.append(f'\\multicolumn{{4}}{{l}}{{\\textbf{{[{part(t)}]}} source: \\texttt{{{tex(t["ciphered_source"])}}}}} \\\\')
    for n in ['4B SFT','8B SFT','8B CoT-GRPO final']:
        m=rows[n][i]; L.append(f'{n} & {tex(m["gold"])} & {hl(m["gold"],m["pred"])} & \\emph{{\\ldots{tex(tail(m["gen"],520))}}} \\\\ \\addlinespace')
    L.append(r'\midrule')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Same held-out Latin items, three models: both SFT models apply a memorised SWELLS template; the CoT-GRPO model reads the case endings.}\label{tab:latin_side}\end{table*}')
open('paper/gen/swells_latin_side.tex','w').write('\n'.join(L))
# item-level agreement
a=rows['8B SFT']; b=rows['8B CoT-GRPO final']
both=sum(1 for x,y in zip(a,b) if x['accuracy']==100 and y['accuracy']==100); onlys=sum(1 for x,y in zip(a,b) if x['accuracy']==100 and y['accuracy']<100); onlyg=sum(1 for x,y in zip(a,b) if x['accuracy']<100 and y['accuracy']==100)
json.dump({'both':both,'only_sft':onlys,'only_grpo':onlyg,'profiles':{n:{g:dict(c) for g,c in d['et'].items()} for n,d in data.items()},'tense_conf':{n:[(k,v) for k,v in d['tense_conf'].most_common(8)] for n,d in data.items()}},open('paper/gen/swells_err.json','w'),indent=1)
print('agreement 8B SFT vs GRPO: both',both,'only sft',onlys,'only grpo',onlyg)
for n,d in data.items(): print(n,d['nfail'],d['tense_conf'].most_common(4))
