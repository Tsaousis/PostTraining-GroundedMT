"""Hus real-language error analysis -> LaTeX fragments (book+dict context)."""
import json,re,random,collections,glob,os
random.seed(5)
O='~/b2s-eval/outputs'
ARMS={'4B base':'qwen3-4b-base','4B SFT':'sft4b','4B GRPO-from-base':'results/real_language/raw/grpo4b-base-full','4B GRPO-from-SFT':'grpo4b-sft','4B forced-CoT GRPO':'grpo4b-forced-full','Gemini 3.1 Pro':'gemini-3.1-pro-preview','8B untrained (thinking)':'~/b2s-eval/outputs_swellsfmt/qwen3-8b','8B CoT-GRPO, SWELLS-only':'~/b2s-eval/outputs_swellsfmt/grpo8b-cot-final','8B CoT-GRPO, MIK (1,998)':'~/b2s-eval/outputs_swellsfmt/grpo8b-real-cot'}
TR={'mizo','ilokano','kimbundu'}
import unicodedata,re as _re
def _nonlat(m):
    try: sc=unicodedata.name(m.group(0)[0]).split()[0].title()
    except ValueError: sc='Other'
    return '['+sc+']'
def tex(s): s=_re.sub(r'[^\x00-\u024f\u2010-\u203a]+',_nonlat,str(s)); return s.replace('\\','').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#').replace('$','\\$').replace('{','\\{').replace('}','\\}').replace('^','')
def loop(h):
    w=h.split(); 
    if len(w)<6: return False
    c=collections.Counter(w); return c.most_common(1)[0][1]>=max(4,len(w)//3)
import sys; sys.path.insert(0,'real_language'); from rescore_hus import clean
import sacrebleu
def load(d):
    rows=[]
    dd=d if d.startswith('/') else f'{O}/{d}'
    for f in glob.glob(f'{dd}/*.jsonl'): rows+= [json.loads(l) for l in open(f)]
    rows=[r for r in rows if r['context']=='book_dict' and not (r['raw'] or '').startswith('__ERROR__')]
    for r in rows: r['hyp']=clean(r['raw'] or '')
    return rows
D={n:load(d) for n,d in ARMS.items()}
# pathology table
L=[r'\begin{table*}[t]\centering\scriptsize\setlength{\tabcolsep}{4pt}\begin{tabular}{l rr rrrr rr rrrr}\toprule',
 r'& \multicolumn{2}{c}{chrF trained} & \multicolumn{4}{c}{unseen en\tr X (1,500)} & \multicolumn{2}{c}{chrF unseen} & \multicolumn{4}{c}{unseen X\tr en (1,500)} \\',
 r'Model & en\tr X & X\tr en & $>$3$\times$ long & copied src & loop & no tag & en\tr X & X\tr en & $>$3$\times$ long & copied & loop & no tag \\ \midrule']
EX=collections.defaultdict(list)
for n,rows in D.items():
    st=collections.defaultdict(list)
    for r in rows:
        u='tr' if r['lang'] in TR else 'un'; d=r['direction']; h=r['hyp'] or ''
        st[(u,d,'chrf')].append(r['chrf'])
        if u=='un':
            st[(u,d,'long')].append(len(h)>3*len(r['reference'])); st[(u,d,'copy')].append(h.strip()==r['source'].strip()); st[(u,d,'loop')].append(loop(h)); st[(u,d,'notag')].append('</translation>' not in (r['raw'] or ''))
            if loop(h): EX[(n,'loop',d)].append(r)
            if h.strip()==r['source'].strip(): EX[(n,'copy',d)].append(r)
            if r['chrf']>45 and not loop(h): EX[(n,'good',d)].append(r)
            if r['chrf']<12 and len(h.split())>4 and not loop(h) and d=='ke': EX[(n,'halluc',d)].append(r)
    def m(k):
        rs=[r for r in rows if ('tr' if r['lang'] in TR else 'un')==k[0] and r['direction']==k[1]]
        return sacrebleu.corpus_chrf([r['hyp'] for r in rs],[[r['reference'] for r in rs]]).score
    L.append(f"{n} & {m(('tr','ek','chrf')):.1f} & {m(('tr','ke','chrf')):.1f} & {sum(st[('un','ek','long')])} & {sum(st[('un','ek','copy')])} & {sum(st[('un','ek','loop')])} & {sum(st[('un','ek','notag')])} & {m(('un','ek','chrf')):.1f} & {m(('un','ke','chrf')):.1f} & {sum(st[('un','ke','long')])} & {sum(st[('un','ke','copy')])} & {sum(st[('un','ke','loop')])} & {sum(st[('un','ke','notag')])} \\\\")
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Real languages, book$+$dict context: corpus chrF (tag-aware extraction, as in Table~\\ref{tab:hus}) and output pathologies on the 15 unseen languages (counts out of 1,500 per direction). ``loop'' $=$ one token makes up $\geq$1/3 of the output; ``copied src'' $=$ output equals the source; ``no tag'' $=$ no closing translation tag (scored on the raw text).}\label{tab:hus_pathology}\end{table*}')
open('paper/naacl/gen/hus_pathology.tex','w').write('\n'.join(L))
# example tables
def row(n,kind,r,cut=150):
    h=(r['hyp'] or '')[:cut]; return f"{n} & {tex(r['lang'])} {r['direction']} & {kind} & {tex(r['source'][:cut])} & {tex(r['reference'][:cut])} & {tex(h)} & {r['chrf']:.0f} \\\\ \\addlinespace"
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{1.7cm}p{1.2cm}p{1cm}p{3.3cm}p{3.3cm}p{3.6cm}r}\toprule','Model & lang & type & source & reference & output & chrF \\\\ \\midrule']
for n in ['4B base','4B SFT','4B GRPO-from-base','4B GRPO-from-SFT','4B forced-CoT GRPO']:
    for kind,d in [('loop','ek'),('copy','ek'),('halluc','ke'),('good','ek'),('good','ke')]:
        xs=EX[(n,kind,d)]
        if xs: L.append(row(n,kind,random.choice(xs)))
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Real-language outputs on unseen languages (book$+$dict), one random example per model and type: ``loop'' $=$ degenerate repetition, ``copy'' $=$ source copied back, ``halluc'' $=$ fluent English unrelated to the source, ``good'' $=$ chrF $>$ 45.}\label{tab:hus_examples}\end{table*}')
open('paper/naacl/gen/hus_examples.tex','w').write('\n'.join(L))
# same sentence across arms (unseen ek + ke), 3 items
by={n:{(r['lang'],r['direction'],r['idx']):r for r in rows} for n,rows in D.items()}
keys=[k for k in by['4B GRPO-from-base'] if k[0] not in TR and k[1]=='ek' and 25<by['4B GRPO-from-base'][k]['chrf']<60]
random.shuffle(keys)
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{2.6cm}p{11cm}r}\toprule']
for k in keys[:3]:
    r=by['4B GRPO-from-base'][k]
    L.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{tex(k[0])}, en\\tr X}} \\quad source: {tex(r['source'][:200])}}} \\\\ \\multicolumn{{3}}{{l}}{{reference: {tex(r['reference'][:200])}}} \\\\ \\midrule")
    for n in ARMS:
        x=by[n].get(k)
        if x: L.append(f"{n} & {tex((x['hyp'] or '')[:220])} & {x['chrf']:.0f} \\\\")
    L.append(r'\addlinespace\midrule')
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{The same unseen-language en\tr X items across all arms. SFT-based arms drift into a trained language or loop; GRPO-from-base produces dictionary-sourced target words; Gemini is the ceiling.}\label{tab:hus_side}\end{table*}')
open('paper/naacl/gen/hus_side.tex','w').write('\n'.join(L))
# forced-CoT traces: one good, one loop, with reasoning tail
F=D['4B forced-CoT GRPO']
def rtail(raw,n=600):
    t=raw.split('<translation>')[0]; t=t.replace('<think>','').replace('</think>',''); return re.sub(r'\s+',' ',t)[-n:]
good=[r for r in F if r['lang'] not in TR and r['direction']=='ek' and r['chrf']>40 and '</translation>' in (r['raw'] or '')]
bad=[r for r in F if r['lang'] not in TR and r['direction']=='ek' and loop(r['hyp'] or '')]
goodke=[r for r in F if r['lang'] not in TR and r['direction']=='ke' and r['chrf']>55 and '</translation>' in (r['raw'] or '')]
L=[r'\begin{table*}[t]\centering\scriptsize\begin{tabular}{p{2cm}p{13.5cm}}\toprule']
for lab,xs in [('good en\\tr X',good),('good X\\tr en',goodke),('loop en\\tr X',bad)]:
    if not xs: continue
    r=random.choice(xs)
    L.append(f"\\textbf{{{lab}}} ({tex(r['lang'])}, chrF {r['chrf']:.0f}) & source: {tex(r['source'][:200])} \\newline reference: {tex(r['reference'][:200])} \\newline reasoning (tail): \\emph{{\\ldots{tex(rtail(r['raw']))}}} \\newline output: {tex((r['hyp'] or '')[:300])} \\\\ \\midrule")
L.append(r'\bottomrule\end{tabular}')
L.append(r'\caption{Forced-CoT traces on unseen languages: the reasoning looks up dictionary entries word by word; in the loop case the model finishes a reasonable plan and then fails to produce the target string.}\label{tab:forced_traces}\end{table*}')
open('paper/naacl/gen/forced_traces.tex','w').write('\n'.join(L))
print('hus done', {k:len(v) for k,v in EX.items() if k[1]=='loop'})
