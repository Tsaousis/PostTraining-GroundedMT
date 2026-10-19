"""Summarise reward_signal_check.py outputs as one comparable row each.

Variance is measured on the extracted <translation> only (reasoning text ignored).
usage: python summarize_cot_signal.py results/reward_signal/cot/*.json [direct.json ...]
"""
import json
import statistics as st
import sys


def row(path):
    d = json.load(open(path))
    gs = d["groups"]
    s = d["summary"]
    n = len(gs[0]["preds"])

    def mean_over(f):
        return st.mean(f(g) for g in gs)

    r = {
        "file": path.split("/")[-1].replace("_signal.json", ""),
        "groups": len(gs),
        "distinct/8": mean_over(lambda g: len(set(g["preds"]))),
        "chrF": s["mean_chrf"],
        "chrF_spread": mean_over(lambda g: max(g["chrf"]) - min(g["chrf"])),
        "zero_spread": mean_over(lambda g: max(g["chrf"]) == min(g["chrf"])),
        "any_exact": s["groups_with_any_exact"],
        "all_exact": s["groups_all_exact"],
        "word_exact": s["mean_word_exact"],
    }
    if "aligned" in gs[0]:
        r["aligned"] = mean_over(lambda g: st.mean(g["aligned"]))
        r["aligned_spread"] = mean_over(lambda g: max(g["aligned"]) - min(g["aligned"]))
    if "raw" in gs[0]:
        r["tag_ok"] = mean_over(lambda g: sum("</translation>" in p for p in g["raw"]) / n)
        r["chars"] = st.mean(len(p) for g in gs for p in g["raw"])
    return r


def main():
    rows = []
    for p in sys.argv[1:]:
        if "groups" in json.load(open(p)):
            rows.append(row(p))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
