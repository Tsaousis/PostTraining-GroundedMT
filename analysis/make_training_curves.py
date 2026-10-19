import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(p, k=8):
    rows=[json.loads(l) for l in open(p)]
    g=[rows[i:i+k] for i in range(0,len(rows)-k+1,k)]
    return (np.array([np.mean([r["score"] for r in b]) for b in g]),
            np.array([np.mean([r["reason_chars"] for r in b]) for b in g])/1000,
            np.array([np.mean([r["has_tag"] for r in b]) for b in g]))
ma=lambda x,w=25: np.convolve(x,np.ones(w)/w,mode="valid")
s1=load("~/plots/data/forced_s1_rewards.jsonl"); s2=load("~/plots/data/forced_s2_rewards.jsonl")
off=len(s1[0])
fig,ax=plt.subplots(1,3,figsize=(7.2,2.2))
ax[0].plot(ma(s1[0]),lw=.8,label="stage 1 (1,998 prompts)"); ax[0].plot(off+np.arange(len(ma(s2[0]))),ma(s2[0]),lw=.8,label="stage 2 (running)")
ax[0].set_title("Forced-CoT GRPO: chrF reward",fontsize=8); ax[0].legend(fontsize=6,frameon=False,loc="lower right"); ax[0].set_xlabel("prompt",fontsize=7)
ax[1].plot(ma(s1[1]),lw=.8,c="C0"); ax[1].plot(off+np.arange(len(ma(s2[1]))),ma(s2[1]),lw=.8,c="C1")
ax[1].plot(ma(s1[2]),":",lw=.8,c="C0"); ax[1].plot(off+np.arange(len(ma(s2[2]))),ma(s2[2]),":",lw=.8,c="C1")
ax[1].set_title("Reasoning k-chars (solid), tag rate (dotted)",fontsize=8); ax[1].set_xlabel("prompt",fontsize=7)
st=json.load(open("~/xfer8b/grpo8b_ep1_adapter/trainer_state.json"))["log_history"]
steps=[h["step"] for h in st if "rewards/chrf_reward/mean" in h]
c=[h["rewards/chrf_reward/mean"] for h in st if "rewards/chrf_reward/mean" in h]
e=[h["rewards/exact_match_diagnostic/mean"] for h in st if "rewards/exact_match_diagnostic/mean" in h]
ax[2].plot(steps[24:],ma(np.array(c)),lw=.8,c="C2",label="chrF reward"); ax[2].plot(steps[24:],ma(np.array(e)),lw=.8,c="C3",label="exact (diagnostic)")
ax[2].set_title("8B CoT GRPO, SWELLS final stage",fontsize=8); ax[2].set_xlabel("step",fontsize=7); ax[2].set_ylim(0,1.05); ax[2].legend(fontsize=6,frameon=False,loc="center right")
for a in ax: a.tick_params(labelsize=7); a.spines[["top","right"]].set_visible(False)
fig.tight_layout(w_pad=1.2); fig.savefig("figures/training_curves.pdf")
