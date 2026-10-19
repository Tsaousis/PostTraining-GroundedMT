"""Rescore saved GRPO-style rollout groups (reward_signal_check.py output) with finer graders.

Graders (all in [0,1]):
  chrf        : sentence chrF (saved)
  exact       : sentence exact (saved)
  word_exact  : fraction of position-aligned words exactly right (saved)
  word_edit   : mean per-aligned-word normalised edit similarity (1 - lev/maxlen)
  bow_exact   : fraction of gold words present anywhere in prediction (order-free)
  order       : normalised Kendall-type score on matched words (1 = same order)
  aligned_word_sim : word-level edit distance with letter-level substitution cost (aligns words first)
  ending      : for aligned word pairs sharing a stem (common prefix >= 3 chars), fraction with identical ending
Mixes: 0.5*chrf + 0.5*g for each g. For every grader we report: within-group spread, how often the group
argmax changes vs chrF, and separation = mean(score correct rollouts) - mean(score wrong rollouts) in groups
that contain both (the quantity GRPO's advantage actually uses).
"""
import json, sys, statistics as st
from itertools import combinations


def norm(s):
    return " ".join(s.lower().replace("@", "").split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_edit(p, g):
    if not p or not g:
        return 0.0
    n = max(len(p), len(g))
    s = sum(1 - lev(a, b) / max(len(a), len(b), 1) for a, b in zip(p, g))
    return s / n


def bow_exact(p, g):
    if not g:
        return 0.0
    pool = list(p)
    hit = 0
    for w in g:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / len(g)


def order(p, g):
    pos = {}
    for i, w in enumerate(p):
        pos.setdefault(w, []).append(i)
    idx = []
    for w in g:
        if pos.get(w):
            idx.append(pos[w].pop(0))
    if len(idx) < 2:
        return 1.0 if len(idx) == len(g) and len(g) <= 1 else 0.0
    conc = sum(1 for i, j in combinations(range(len(idx)), 2) if idx[i] < idx[j])
    return conc / (len(idx) * (len(idx) - 1) / 2)


def ending(p, g, stem_len=3):
    pairs = [(a, b) for a, b in zip(p, g) if a[:stem_len] == b[:stem_len] and len(b) >= stem_len]
    if not pairs:
        return 0.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def aligned_word_sim(p, g):
    """Word-level edit distance where substitution costs the letter-level distance of the pair
    (insert/delete cost 1). Score = 1 - cost / max(len(p), len(g)). Order-aware, no double penalty."""
    if not p or not g:
        return 0.0
    prev = [float(j) for j in range(len(g) + 1)]
    for i, a in enumerate(p, 1):
        cur = [float(i)]
        for j, b in enumerate(g, 1):
            sub = prev[j - 1] + lev(a, b) / max(len(a), len(b), 1)
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, sub))
        prev = cur
    return max(0.0, 1 - prev[-1] / max(len(p), len(g)))


def main(path):
    d = json.load(open(path))
    groups = d["groups"]
    names = ["chrf", "exact", "word_exact", "word_edit", "aligned_word_sim", "bow_exact", "order", "ending"]
    scored = []
    for grp in groups:
        gw = norm(grp["gold"]).split()
        rows = []
        for k, pred in enumerate(grp["preds"]):
            pw = norm(pred).split()
            rows.append({
                "chrf": grp["chrf"][k], "exact": grp["exact"][k], "word_exact": grp["word_exact"][k],
                "word_edit": word_edit(pw, gw), "bow_exact": bow_exact(pw, gw),
                "aligned_word_sim": aligned_word_sim(pw, gw),
                "order": order(pw, gw), "ending": ending(pw, gw),
            })
        scored.append(rows)
    out = {}
    for g in names:
        spread, changed, sep, mix_changed = [], 0, [], 0
        for rows in scored:
            v = [r[g] for r in rows]
            spread.append(max(v) - min(v))
            c = [r["chrf"] for r in rows]
            best_c = max(range(len(c)), key=lambda i: c[i])
            if max(range(len(v)), key=lambda i: v[i]) != best_c and v[best_c] < max(v):
                changed += 1
            mix = [0.5 * a + 0.5 * b for a, b in zip(c, v)]
            if max(range(len(mix)), key=lambda i: mix[i]) != best_c and mix[best_c] < max(mix):
                mix_changed += 1
            ok = [r[g] for r in rows if r["exact"] == 1.0]
            bad = [r[g] for r in rows if r["exact"] == 0.0]
            if ok and bad:
                sep.append(st.mean(ok) - st.mean(bad))
        out[g] = {
            "mean": round(st.mean(r[g] for rows in scored for r in rows), 3),
            "mean_group_spread": round(st.mean(spread), 3),
            "frac_groups_spread0": round(sum(1 for s in spread if s == 0) / len(spread), 3),
            "argmax_changed_vs_chrf": round(changed / len(scored), 3),
            "argmax_changed_mix_vs_chrf": round(mix_changed / len(scored), 3),
            "sep_correct_minus_wrong(mixed groups,n=%d)" % len(sep): round(st.mean(sep), 3) if sep else None,
        }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
