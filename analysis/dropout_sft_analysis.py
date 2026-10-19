"""Grammar-dropout SFT vs plain curriculum SFT (Qwen3-4B-Base), matched 4k-cap evals.
Per-partition exact match in three test contexts (W+G, dictionary-only, task-only) and
spurious grammar citations in dictionary-only traces."""
import json, re, sys, collections
from pathlib import Path
R = Path.home() / "swells-thesis/results"
SRC = {
 ("plain","wg"): Path.home()/"results/sft_curriculum_wg_4k",
 ("plain","w"): R/"raw/sft4b_w_task_4k", ("plain","task"): R/"raw/sft4b_w_task_4k",
 ("drop","wg"): R/"raw/sft_dropout/sft_dropout_grammar_excerpts_4k",
 ("drop","w"): R/"raw/sft_dropout/sft_dropout_dictionary_only_4k",
 ("drop","task"): R/"raw/sft_dropout/sft_dropout_task_only_4k",
}
CTX = {"wg":"grammar_excerpts","w":"dictionary_only","task":"task_only"}
def part(ph): return ph.split(".")[0]
rows = collections.defaultdict(dict); overall = {}
for (arm,ctx),d in SRC.items():
    s = json.load(open(d/"summary.json"))
    for direc in ("art_eng","eng_art"):
        k = f"CoT_full_{direc}_test_{CTX[ctx]}"
        if k not in s: continue
        v = s[k]; overall[(arm,ctx,direc)] = (v["n"], v["chrf"], v["accuracy"])
        agg = collections.defaultdict(lambda:[0,0])
        for ph,x in v["by_phenomenon"].items():
            p = part(ph); agg[p][0] += x["accuracy"]*x["n"]/100; agg[p][1] += x["n"]
        for p,(c,n) in agg.items(): rows[(direc,p)][(arm,ctx)] = 100*c/n
order = ["1","2m","2f","3","4","5m","5f","6m","6f","7m","7f","8a","8g","9a","9g"]
cols = [(a,c) for c in ("wg","w","task") for a in ("plain","drop")]
with open(R/"analysis/dropout/per_partition.tsv","w") as f:
    f.write("direction\tpartition\t" + "\t".join(f"{a}_{c}" for a,c in cols) + "\n")
    for direc in ("art_eng","eng_art"):
        for p in order:
            if (direc,p) not in rows: continue
            f.write(f"{direc}\t{p}\t" + "\t".join(f"{rows[(direc,p)].get(k,float('nan')):.1f}" for k in cols) + "\n")
with open(R/"analysis/dropout/overall.tsv","w") as f:
    f.write("arm\tcontext\tdirection\tn\tchrf\texact\n")
    for (a,c,dr),(n,ch,ac) in sorted(overall.items()): f.write(f"{a}\t{c}\t{dr}\t{n}\t{ch:.1f}\t{ac:.1f}\n")
# spurious grammar citations + cap hits in dictionary-only / task-only traces
CITE = re.compile(r"grammar (book|rule|excerpt)|according to the grammar|the grammar (says|states|indicates)", re.I)
with open(R/"analysis/dropout/trace_stats.tsv","w") as f:
    f.write("arm\tcontext\tdirection\tn\tcites_grammar_pct\tmentions_dictionary_pct\tmean_gen_chars\thas_tag_pct\n")
    for (arm,ctx),d in SRC.items():
        for direc in ("art_eng","eng_art"):
            p = d/f"CoT_full_{direc}_test_{CTX[ctx]}.jsonl"
            if not p.exists(): continue
            gens = [json.loads(l)["gen"].split("</translation>")[0] for l in open(p)]
            n=len(gens)
            f.write(f"{arm}\t{ctx}\t{direc}\t{n}\t{100*sum(bool(CITE.search(g)) for g in gens)/n:.1f}\t"
                    f"{100*sum('dictionar' in g.lower() for g in gens)/n:.1f}\t{sum(map(len,gens))/n:.0f}\t"
                    f"{100*sum('<translation>' in g for g in gens)/n:.1f}\n")
for fn in ("overall.tsv","per_partition.tsv","trace_stats.tsv"): print(open(R/"analysis/dropout"/fn).read())
