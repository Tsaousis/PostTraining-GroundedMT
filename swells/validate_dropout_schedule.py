"""Sanity checks for a build_dropout_schedule.py output directory against the frozen schedule."""
import argparse
import json
import re
import sys

STAGES = [f"course_{i}" for i in range(1, 8)]


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frozen", default="${SWELLS_SCHEDULE:-data/swells_schedule}")
    ap.add_argument("--final_epochs", type=int, default=4)
    a = ap.parse_args()
    stages = STAGES + [f"final_ep{i}" for i in range(1, a.final_epochs + 1)]
    ok = True
    last_p = -1.0
    for st in stages:
        frozen = load(f"{a.frozen}/{'final_1408' if st.startswith('final') else st}.jsonl")
        cot = load(f"{a.out}/cot/{st}.jsonl")
        nocot = load(f"{a.out}/nocot/{st}.jsonl")
        meta = load(f"{a.out}/meta/{st}.jsonl")
        n = len(frozen)
        same_target = sum(c["messages"][2]["content"] == frozen[m["frozen_index"]]["messages"][2]["content"] for c, m in zip(cot, meta))
        identical_prompt = sum(c["messages"][1]["content"] == frozen[m["frozen_index"]]["messages"][1]["content"] for c, m in zip(cot, meta))
        perm_ok = sorted(m["frozen_index"] for m in meta) == list(range(n))
        ps = [m["p"] for m in meta]
        mono = all(x >= y for x, y in zip(ps[1:], ps)) and ps[0] >= last_p
        last_p = ps[-1]
        no_gram = sum("Grammar Book Excerpts" not in c["messages"][1]["content"] for c in cot)
        has_dict = sum("dictionary entries" in c["messages"][1]["content"] for c in cot)
        kept_frac = sum(m["kept_needed"] for m in meta) / max(1, sum(m["n_needed"] for m in meta))
        dis_frac = sum(m["n_distractor"] for m in meta) / max(1, sum(m["n_needed"] for m in meta))
        nocot_ok = sum(
            re.search(r"<translation>(.*?)</translation>", c["messages"][2]["content"], re.S).group(1).strip()
            == re.search(r"<translation>(.*?)</translation>", nc["messages"][2]["content"], re.S).group(1).strip()
            and "Do not explain" in nc["messages"][1]["content"]
            for c, nc in zip(cot, nocot)
        )
        maxtok = max(m.get("prompt_tokens", 0) for m in meta)
        line = (
            f"{st:9s} n={len(cot)}/{n} nocot={len(nocot)} meta={len(meta)} targets_same={same_target} "
            f"prompt_identical={identical_prompt} perm_ok={perm_ok} p={ps[0]:.3f}->{ps[-1]:.3f} mono={mono} "
            f"kept_needed={kept_frac:.2f} distractor={dis_frac:.2f} no_grammar={no_gram} dict={has_dict} "
            f"nocot_ok={nocot_ok} max_tok={maxtok}"
        )
        print(line)
        bad = (
            len(cot) != n or len(nocot) != n or len(meta) != n or same_target != n or not mono
            or has_dict != n or nocot_ok != n or not perm_ok
        )
        if st == "course_1" and ps[0] != 0.0:
            bad = True
        if ps[0] >= 1.0 and no_gram != n:
            bad = True
        if bad:
            ok = False
            print("  ^^^ FAIL")
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
