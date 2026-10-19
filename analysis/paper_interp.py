"""Interpretability analyses for the paper (no GPU):
 1. SWELLS exact accuracy by grammatical feature x arm (+ logistic regression of correctness on features)
 2. lexical-vs-grammar credit split: P(exact)=P(lexicon right)*P(exact|lexicon right), SWELLS + Hus X->en
 3. trace grounding vs correctness for CoT arms (8B SWELLS, forced-CoT Hus, Gemini Hus)
 4. paired bootstrap for 8B SWELLS GRPO vs SFT
Writes results/analysis/interp/*.tsv and paper/naacl/gen/interp_*.tex"""
import json,re,random,collections,sys,os,glob
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,'real_language')
from rescore_hus import clean,TR
from sacrebleu.metrics import CHRF
_C=CHRF()
random.seed(0); np.random.seed(0)
OUT='results/analysis/interp'; GEN='paper/gen'
R='results'
TEST='~/swells_min/SWELLS_Dataset/Test/Test_CoT/CoT_full_art_eng_test_grammar_excerpts.jsonl'
T=[json.loads(l) for l in open(TEST)]
M={'4B base':'~/results/base4b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B SFT':'~/results/sft_curriculum_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '4B GRPO no-CoT':f'{R}/raw/grpo_base_ep3/grpo_base_ep3_wg_nocot_4k/no_CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B base':'~/results/base8b_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B SFT':f'{R}/raw/sft8b_think_wg/CoT_full_art_eng_test_grammar_excerpts.jsonl',
 '8B CoT-GRPO':f'{R}/raw/grpo8b_cot_final_wg_4k/CoT_full_art_eng_test_grammar_excerpts.jsonl'}
D={n:[json.loads(l) for l in open(f)] for n,f in M.items()}
for n,r in D.items(): assert len(r)==len(T)==1400,n
acc={n:np.array([float(x['accuracy'])/100 for x in r]) for n,r in D.items()}
def tex(s): return s.replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#')
def boot_mean(d,B=2000):
    d=np.asarray(d); bs=np.sort([np.random.choice(d,len(d)).mean() for _ in range(B)]); return d.mean(),bs[int(.025*B)],bs[int(.975*B)-1]
# ---------- features ----------
def feats(t):
    it=t['instance_type']; head=it.split('___')[0]; cipher=it.split('___')[-1]
    p=head.split('.')[0]; part='6' if p.startswith('6') else p
    f={'partition':part,'cipher':cipher,
       'noun':(re.search(r'\bn([ABC])\b',head.replace('_','.')) or [None,'-'])[1],
       'verb':(re.search(r'\bv(A|C|GEN)\b',head.replace('_','.')) or [None,'-'])[1],
       'ending':(re.search(r'\be([AC])\b',head.replace('_','.')) or [None,'-'])[1],
       'adj':(re.search(r'\badj([ABC])\b',head.replace('_','.')) or [None,'-'])[1],
       'adjorder':(re.search(r'\badjO([ABC])\b',head) or [None,'-'])[1],
       'person':(re.search(r'\bpe([AC])\b',head.replace('_','.')) or [None,'-'])[1],
       'gender':'f' if re.search(r'\.f\.|^[257]f',head) else ('m' if re.search(r'\.m\.|^[257]m',head) else '-'),
       'latin':'yes' if part in('8a','8g','9a','9g') else 'no',
       'nwords':str(min(len(t['reference'].split()),6))}
    return f
F=[feats(t) for t in T]
FEAT_ORDER=['partition','cipher','noun','verb','ending','adj','adjorder','person','gender','nwords']
ARMS=list(M)
rows=[]
for fn in FEAT_ORDER:
    vals=sorted({f[fn] for f in F})
    for v in vals:
        idx=[i for i,f in enumerate(F) if f[fn]==v]
        if len(idx)<20 or v=='-': continue
        rows.append([fn,v,len(idx)]+[100*acc[a][idx].mean() for a in ARMS])
with open(f'{OUT}/swells_feature_acc.tsv','w') as f:
    f.write('feature\tvalue\tn\t'+'\t'.join(ARMS)+'\n')
    for r in rows: f.write('\t'.join([r[0],r[1],str(r[2])]+['%.1f'%x for x in r[3:]])+'\n')
# which features matter after controlling for the others: per-arm logistic regression on one-hot (non-Latin items only, to avoid the all-zero Latin block)
NL=[i for i,f in enumerate(F) if f['latin']=='no']
cols=[]; X=[]
for fn in ['cipher','noun','verb','ending','adj','adjorder','person','gender']:
    vals=sorted({F[i][fn] for i in NL}); vals=[v for v in vals if v!='-']
    for v in vals[1:]: cols.append(f'{fn}={v} (vs {vals[0]})'); X.append([1.0 if F[i][fn]==v else 0.0 for i in NL])
X=np.array(X).T
with open(f'{OUT}/swells_feature_logit.tsv','w') as f:
    f.write('coef (log-odds, partitions 1-7 only)\t'+'\t'.join(ARMS)+'\n')
    co={}
    for a in ARMS:
        y=acc[a][NL]
        if y.min()==y.max(): co[a]=[float('nan')]*len(cols); continue
        co[a]=LogisticRegression(C=1.0,max_iter=2000).fit(X,y).coef_[0]
    for j,c in enumerate(cols): f.write(c+'\t'+'\t'.join('%.2f'%co[a][j] for a in ARMS)+'\n')
# ---------- 2. lexical vs grammar split ----------
FUNC={'the','a','an','i','you','he','she','it','we','they','his','her','their','my','your','our','him','them','us','me','will','would','was','were','is','are','am','had','has','have','been','being','did','do','does','to','of','in','on','at','by','for','with','and','or','not','that','this','these','those','be','there','who','which','as','from','into','than','so','but','if'}
def words(s): return re.findall(r"[a-z']+",s.lower())
def content(s): return [w for w in words(s) if w not in FUNC]
def stem(w): return w[:4]
def lex_ok(gold,pred):
    g={stem(w) for w in content(gold)}; p={stem(w) for w in content(pred)}
    return float(g<=p and len(p-g)<=0)  # all content stems present, none extra
lex={n:np.array([lex_ok(x['gold'],x['pred']) for x in r]) for n,r in D.items()}
with open(f'{OUT}/swells_lex_split.tsv','w') as f:
    f.write('arm\tgroup\tn\tP(lexicon right)\tP(exact | lexicon right)\tP(exact | lexicon wrong)\tP(exact)\n')
    for a in ARMS:
        for g,idx in [('1-5',[i for i,ff in enumerate(F) if ff['partition'] in('1','2f','2m','3','4','5f','5m')]),('6-7',[i for i,ff in enumerate(F) if ff['partition'] in('6','7f','7m')]),('Latin',[i for i,ff in enumerate(F) if ff['latin']=='yes']),('all',list(range(1400)))]:
            l=lex[a][idx]; y=acc[a][idx]
            f.write('\t'.join([a,g,str(len(idx)),'%.1f'%(100*l.mean()),'%.1f'%(100*y[l==1].mean() if (l==1).any() else float('nan')),'%.1f'%(100*y[l==0].mean() if (l==0).any() else float('nan')),'%.1f'%(100*y.mean())])+'\n')
# Hus X->en: content-word F1 vs function-word F1 (book_dict, 15 unseen languages)
HUS={'4B base':'~/b2s-eval/outputs/qwen3-4b-base','4B SFT':'~/b2s-eval/outputs/sft4b',
     '4B GRPO-from-base':f'{R}/real_language/raw/grpo4b-base-full','4B GRPO-from-SFT':'~/b2s-eval/outputs/grpo4b-sft',
     '4B forced-CoT s1':'~/b2s-eval/outputs/grpo4b-forced-s1',
     '8B untrained':f'{R}/real_language/hus_outputs_swellsfmt/qwen3-8b','8B SWELLS CoT-GRPO':f'{R}/real_language/hus_outputs_swellsfmt/grpo8b-cot-final',
     'Gemini 3.1 Pro':'~/b2s-eval/outputs/gemini-3.1-pro-preview'}
def load_hus(d):
    rows=[]
    for fl in glob.glob(f'{d}/*.jsonl'):
        for l in open(fl):
            r=json.loads(l)
            if r.get('context')!='book_dict' or (r.get('raw') or '').startswith('__ERROR__'): continue
            r['hyp']=clean(r.get('raw') or ''); rows.append(r)
    return rows
H={n:load_hus(d) for n,d in HUS.items()}
print({n:(len(r),collections.Counter(x['direction'] for x in r)) for n,r in H.items()})
def f1(a,b):
    ca,cb=collections.Counter(a),collections.Counter(b); ov=sum((ca&cb).values())
    if not a or not b: return 0.0
    p,r=ov/len(b),ov/len(a); return 2*p*r/(p+r) if p+r else 0.0
def xen(r): return r['direction'].endswith('e') and not r['direction'].startswith('e')
with open(f'{OUT}/hus_lex_split.tsv','w') as f:
    f.write('arm\tlangs\tn\tcontent-word F1\tfunction-word F1\tstem F1 (content)\tsentence chrF\n')
    for a,rows in H.items():
        for g,sel in [('unseen15',[r for r in rows if xen(r) and r['lang'] not in TR]),('trained3',[r for r in rows if xen(r) and r['lang'] in TR])]:
            if not sel: continue
            cf=np.mean([f1(content(r['reference']),content(r['hyp'])) for r in sel]); ff=np.mean([f1([w for w in words(r['reference']) if w in FUNC],[w for w in words(r['hyp']) if w in FUNC]) for r in sel])
            sf=np.mean([f1([stem(w) for w in content(r['reference'])],[stem(w) for w in content(r['hyp'])]) for r in sel])
            ch=np.mean([_C.sentence_score(r['hyp'],[r['reference']]).score for r in sel])
            f.write('\t'.join([a,g,str(len(sel)),'%.1f'%(100*cf),'%.1f'%(100*ff),'%.1f'%(100*sf),'%.1f'%ch])+'\n')
# ---------- 3. trace grounding vs correctness ----------
PAT={'dict':r'\b(dictionar|entry|entries|lexicon|word list|vocabulary)','rule':r'\b(grammar|rule|declension|conjugat|suffix|prefix|ending|case|agreement|plural|tense|aspect|passive|genitive|nominative|accusative)',
     'quote':r'["“”][^"“”\n]{2,60}["“”]|\'[^\'\n]{2,40}\'\s*(means|=|→|->|is)','hedge':r'\b(maybe|perhaps|not sure|unclear|guess|assume|probably|might be|possibly|I think)','example':r'\b(example|e\.g\.|for instance|the given|provided (sentence|pair|translation)s?)'}
def trace_feats(t):
    t=t or ''; return {k:float(bool(re.search(p,t,re.I))) for k,p in PAT.items()}|{'loglen':np.log1p(len(t))}
def think_of(gen):
    m=re.search(r'<think>(.*?)(</think>|$)',gen or '',re.S); return m.group(1) if m else ''
sets={}
sets['8B CoT-GRPO (SWELLS, exact)']=[(trace_feats(think_of(x['gen'])),float(x['accuracy'])/100) for x in D['8B CoT-GRPO']]
sets['8B SFT thinking (SWELLS, exact)']=[(trace_feats(think_of(x['gen'])),float(x['accuracy'])/100) for x in D['8B SFT']]
sets['4B forced-CoT s1 (Hus unseen X->en, chrF/100)']=[(trace_feats(think_of(r['raw'])),_C.sentence_score(r['hyp'],[r['reference']]).score/100) for r in H['4B forced-CoT s1'] if xen(r) and r['lang'] not in TR and '</think>' in (r['raw'] or '')]
sets['8B SWELLS CoT-GRPO (Hus unseen X->en, chrF/100)']=[(trace_feats(think_of(r['raw'])),_C.sentence_score(r['hyp'],[r['reference']]).score/100) for r in H['8B SWELLS CoT-GRPO'] if xen(r) and r['lang'] not in TR and '</think>' in (r['raw'] or '')]
gem=[]
for fl in glob.glob(f'{R}/real_language/gemini_cot/raw/*.jsonl'):
    for l in open(fl):
        r=json.loads(l)
        if r.get('answer','').startswith('__ERROR__') or (r['arm']=='cot' and '<translation>' not in r['answer']): continue
        hyp=clean(r['answer']); gem.append((r['arm'],r['lang'],trace_feats(r['thoughts']+('\n'+r['answer'] if r['arm']=='cot' else '')),_C.sentence_score(hyp,[r['reference']]).score/100))
for arm in ['think','cot']:
    sets[f'Gemini {arm} (Hus 3 langs, chrF/100)']=[(f,y) for a,l,f,y in gem if a==arm]
    sets[f'Gemini {arm} Kalamang only (chrF/100)']=[(f,y) for a,l,f,y in gem if a==arm and l=='kalamang']
with open(f'{OUT}/trace_grounding.tsv','w') as f:
    f.write('set\tn\tfeature\tprevalence%\toutcome|feature\toutcome|no feature\tdiff\tCI95\n')
    for name,S in sets.items():
        y=np.array([s[1] for s in S])
        for k in PAT:
            x=np.array([s[0][k] for s in S])
            if x.min()==x.max(): f.write(f'{name}\t{len(S)}\t{k}\t{100*x.mean():.0f}\t-\t-\t-\t-\n'); continue
            a,b=y[x==1],y[x==0]; d=a.mean()-b.mean()
            bs=np.sort([np.random.choice(a,len(a)).mean()-np.random.choice(b,len(b)).mean() for _ in range(2000)])
            f.write(f'{name}\t{len(S)}\t{k}\t{100*x.mean():.0f}\t{100*a.mean():.1f}\t{100*b.mean():.1f}\t{100*d:+.1f}\t[{100*bs[50]:+.1f},{100*bs[1949]:+.1f}]\n')
        # joint logistic (binary outcome: exact, or chrF>=median)
        Xf=np.array([[s[0][k] for k in list(PAT)+['loglen']] for s in S]); yb=(y>=np.median(y)).astype(float) if y.max()>1-1e-9 and set(np.unique(y))-{0.0,1.0} else y
        if yb.min()!=yb.max():
            co=LogisticRegression(max_iter=2000).fit(Xf,yb).coef_[0]
            f.write(f'{name}\t{len(S)}\tlogit(all feats + loglen)\t\t'+' '.join(f'{k}={c:+.2f}' for k,c in zip(list(PAT)+['loglen'],co))+'\t\t\t\n')
# ---------- 4. 8B bootstrap ----------
groups=collections.defaultdict(list)
for i,ff in enumerate(F): groups['all'].append(i); groups['Latin' if ff['latin']=='yes' else ('6-7' if ff['partition'] in('6','7f','7m') else '1-5')].append(i)
with open(f'{OUT}/swells_8b_bootstrap.tsv','w') as f:
    f.write('comparison\tgroup\tn\tmean diff (acc pts)\tCI95 lo\tCI95 hi\n')
    for a,b in [('8B CoT-GRPO','8B SFT'),('8B CoT-GRPO','8B base'),('8B SFT','8B base'),('4B GRPO no-CoT','4B SFT'),('4B SFT','4B base'),('4B GRPO no-CoT','4B base')]:
        for g in ['all','1-5','6-7','Latin']:
            idx=groups[g]; m,lo,hi=boot_mean(acc[a][idx]-acc[b][idx]); f.write(f'{a} - {b}\t{g}\t{len(idx)}\t{100*m:+.1f}\t{100*lo:+.1f}\t{100*hi:+.1f}\n')
print('done')
