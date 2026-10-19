import json,numpy as np,matplotlib,collections,re; matplotlib.use('Agg'); import matplotlib.pyplot as plt
G='paper/gen/'; F='figures/'
def win(xs,k=40):
    xs=np.array(xs,float); n=len(xs)//k; return np.arange(n)*k+k/2, xs[:n*k].reshape(n,k).mean(1)
fig,ax=plt.subplots(2,4,figsize=(14.5,5.6))
# (a) 8B CoT-GRPO ep1 trainer state
h=json.load(open('~/xfer8b/grpo8b_ep1_adapter/trainer_state.json'))['log_history']
s=[x['step'] for x in h if 'reward' in x]; r=[x['reward'] for x in h if 'reward' in x]; e=[x['rewards/exact_match_diagnostic/mean'] for x in h if 'reward' in x]; c=[x['completions/clipped_ratio'] for x in h if 'reward' in x]
a=ax[0,0]; sm=lambda v:np.convolve(v,np.ones(25)/25,'valid'); ss=s[12:12+len(sm(r))]; a.plot(ss,sm(r),label='chrF reward'); a.plot(ss,sm(e),label='exact (diagnostic)'); a.plot(ss,sm(c),label='clipped ratio'); a.set_title('(a) 8B CoT-GRPO, final stage ep.1'); a.legend(fontsize=7); a.set_xlabel('step')
# (b) grader-v2 pilot B 8B
rows=[json.loads(l) for l in open('~/pilotb/runs/grpo_v2_pilot_course4_qwen8b_instruct_cot/course_4/rewards_v2.jsonl')]
a=ax[0,1]
for k in ['total','lex','rule','order','fmt','chrf']:
    x,y=win([r[k] for r in rows],80); a.plot(x,y,label=k)
a.set_title('(b) grader-v2 pilot, 8B instruct, course 4'); a.legend(fontsize=7,ncol=2); a.set_xlabel('rollout')
# (c) grader-v2 4B pilots
a=ax[0,2]
for tag,f in [('no-CoT','nocot'),('CoT','cot')]:
    rows=[json.loads(l) for l in open(f'results/pilot_v2/{f}_course4_rewards_v2.jsonl')]
    for k,ls in [('total','-'),('rule',':'),('chrf','--')]:
        x,y=win([r[k] for r in rows],80); a.plot(x,y,ls,label=f'{tag} {k}')
a.set_title('(c) grader-v2 pilots, 4B base, course 4'); a.legend(fontsize=7,ncol=2); a.set_xlabel('rollout')
# (d) 4B CoT chrF-GRPO final stage (cot2_logs)
rows=[json.loads(l) for l in open('~/cot2_logs/rewards_final.jsonl')]
a=ax[1,0]; x,y=win([r['score'] for r in rows],96); a.plot(x,y,label='chrF reward'); x,y=win([r['has_tag'] for r in rows],96); a.plot(x,y,label='has tag'); x,y=win([min(r['gen_chars'],6000)/6000 for r in rows],96); a.plot(x,y,label='gen chars /6k'); a.set_title('(d) 4B CoT chrF-GRPO, final stage'); a.legend(fontsize=7); a.set_xlabel('rollout')
# (e) forced-CoT real languages
a=ax[1,1]
off=0
for tag,f in [('stage 1','forced_s1'),('stage 2','forced_s2')]:
    rows=[json.loads(l) for l in open(f'~/plots/data/{f}_rewards.jsonl')]
    k=[k for k in ('score','reward','chrf') if k in rows[0]][0]
    x,y=win([r[k] for r in rows],64); a.plot(x+off,y,label=f'{tag} reward'); off+=len(rows)
a.set_title('(e) forced-CoT GRPO, 4B instruct, MIK'); a.legend(fontsize=7); a.set_xlabel('rollout')
# (f) word-exact pilot vs chrF run first 100 steps
st=json.load(open('results/reward_signal/pilot_direct_wordexact_c7/trainer_state_step100.json'))['log_history']
a=ax[1,2]
ks=[k for k in st[0] if k.startswith('rewards/') and k.endswith('/mean')]
for k in ks:
    v=np.array([x[k] for x in st if k in x]); a.plot([x['step'] for x in st if k in x][9:9+len(v)-18],np.convolve(v,np.ones(19)/19,'valid'),label=k.split('/')[1]+' (19-step mean)')
try:
    ch=json.load(open('results/reward_signal/chrf_run_final_steps1-120.json'))
    if isinstance(ch,list) and ch and 'reward' in ch[0]: v=np.array([x['reward'] for x in ch]); a.plot([x['step'] for x in ch][9:9+len(v)-18],np.convolve(v,np.ones(19)/19,'valid'),'k--',label='plain chrF run')
except Exception as ex: print('chrf run',ex)
a.set_title('(f) chrF+word-exact pilot, 4B course 7'); a.legend(fontsize=7); a.set_xlabel('step')
# (g) 8B CoT-GRPO on real languages (MIK, 1,998 prompts)
a=ax[0,3]; rows=[json.loads(l) for l in open('~/plots/data/grpo8b_real_rewards.jsonl')]
x,y=win([r['score'] or 0 for r in rows],96); a.plot(x,y,label='chrF reward'); x,y=win([bool(r['has_tag']) for r in rows],96); a.plot(x,y,label='has tag'); x,y=win([min(r['reason_chars'] or 0,6000)/6000 for r in rows],96); a.plot(x,y,label='reasoning chars /6k')
a.set_title('(g) CoT-GRPO, 8B instruct, MIK'); a.legend(fontsize=7); a.set_xlabel('rollout')
ax[1,3].axis('off')
plt.tight_layout(); plt.savefig(F+'reward_curves.pdf')
# reward hacking examples from pilot B: high total, low chrF / wrong tense
rows=[json.loads(l) for l in open('~/pilotb/runs/grpo_v2_pilot_course4_qwen8b_instruct_cot/course_4/rewards_v2.jsonl')]
late=rows[-800:]
def tex(s): return str(s).replace('\\','').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#').replace('$','\\$').replace('{','\\{').replace('}','\\}')
hack=sorted([r for r in late if r['total']>0.6 and r['chrf']<0.45],key=lambda r:-r['total'])[:4]
L=[r'\begin{table}[t]\centering\scriptsize\begin{tabular}{p{2.6cm}p{2.6cm}rrrr}\toprule','gold & prediction & total & lex & rule & chrF \\\\ \\midrule']
for r in hack: L.append(f"{tex(r['gold'])} & {tex(r['pred'][:80])} & {r['total']:.2f} & {r['lex']:.2f} & {r['rule']:.2f} & {r['chrf']:.2f} \\\\")
L.append(r'\bottomrule\end{tabular}\caption{Grader-v2 pilot (8B, course 4, last 800 rollouts): rollouts with high grader total but low chrF -- the lexical and format terms are satisfied while the string is wrong.}\label{tab:hack}\end{table}')
open(G+'reward_hack.tex','w').write('\n'.join(L))
# length stats pilot v2 nocot 4B: one word short
for f in ['nocot','cot']:
    rows=[json.loads(l) for l in open(f'results/pilot_v2/{f}_course4_rewards_v2.jsonl')]
    print(f,'mean pred words',np.mean([len(r['pred'].split()) for r in rows[-600:]]),'gold',np.mean([len(r['gold'].split()) for r in rows[-600:]]))
print('figs done')
