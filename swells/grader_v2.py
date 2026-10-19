"""Grader v2: rule-based decomposed reward for SWELLS translations.

Inputs per instance: gold target string and the target-side dictionary lemmas
(the rule-OFF forms: what a dictionary lookup alone would produce). Gold tokens
that align to a lemma are content tokens; the character diff between a content
token and its lemma is the rule's diagnostic feature (affix / agreement /
mutation). Gold tokens with no lemma are function words introduced by the
grammar (articles, pronouns, auxiliaries).

Components, each in [0, 1]:
  lex    fraction of content tokens whose lexeme appears in the prediction
         (fuzzy match on the lemma, so a wrong affix still counts as right word)
  rule   fraction of gold tokens whose exact surface form appears in the
         prediction; a content token gets 0.5 if it has the lemma's stem and
         some affix change but not the right one, 0 if left in rule-OFF form
  order  LCS of matched tokens over len(gold) (word order / syntax)
  fmt    1 if a <translation> tag was emitted
Total = weighted sum. Default weights drop the free format credit.
"""
import re
from difflib import SequenceMatcher

TAG = re.compile(r"<translation>(.*?)</translation>", re.S)
DEFAULT_WEIGHTS = {"lex": 0.3, "rule": 0.5, "order": 0.2, "fmt": 0.0}


def norm(text):
    return (text or "").strip().strip("@").strip().lower().rstrip(".").strip()


def tokens(text):
    return [t for t in re.split(r"[\s']+", norm(text)) if t]


def sim(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def stem(a, b):
    """Longest common prefix or suffix between a lemma and its inflected form."""
    m = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return a[m.a : m.a + m.size]


def align_gold_to_lemmas(gold_toks, lemmas, thr=0.6):
    """Greedy one-to-one alignment of gold tokens to lemmas by similarity."""
    pairs = sorted(
        ((sim(g, l), gi, li) for gi, g in enumerate(gold_toks) for li, l in enumerate(lemmas)),
        reverse=True,
    )
    g2l, used = {}, set()
    for s, gi, li in pairs:
        if s < thr:
            break
        if gi in g2l or li in used:
            continue
        g2l[gi] = li
        used.add(li)
    return g2l


def lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[len(a)][len(b)]


def grade(output, gold, lemmas, weights=None, lemma_thr=0.6, lex_thr=0.6):
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    m = TAG.findall(output or "")
    fmt = 1.0 if m else 0.0
    if m:
        pred = m[-1]
    else:
        lines = [l for l in (output or "").splitlines() if l.strip()]
        pred = lines[-1] if lines else ""
    g = tokens(gold)
    p = tokens(pred)
    lemmas = [norm(l) for l in lemmas if norm(l)]
    if not g:
        return {"lex": 0.0, "rule": 0.0, "order": 0.0, "fmt": fmt, "total": 0.0}
    if not p:
        return {"lex": 0.0, "rule": 0.0, "order": 0.0, "fmt": fmt, "total": w["fmt"] * fmt}

    g2l = align_gold_to_lemmas(g, lemmas, lemma_thr)

    # match each gold token to at most one prediction token
    used_p = set()
    match = {}  # gi -> (pi, kind) kind: exact | lexeme | none
    for gi, gt in enumerate(g):
        for pi, pt in enumerate(p):
            if pi not in used_p and pt == gt:
                match[gi] = (pi, "exact")
                used_p.add(pi)
                break
    for gi, gt in enumerate(g):
        if gi in match or gi not in g2l:
            continue
        lemma = lemmas[g2l[gi]]
        st = stem(lemma, gt)
        best, best_pi = 0.0, None
        for pi, pt in enumerate(p):
            if pi in used_p:
                continue
            s = max(sim(pt, lemma), sim(pt, gt))
            if (st and st in pt) or s >= lex_thr:
                if s > best:
                    best, best_pi = s, pi
        if best_pi is not None:
            match[gi] = (best_pi, "lexeme")
            used_p.add(best_pi)

    content = [gi for gi in range(len(g)) if gi in g2l]
    lex_hits = sum(1 for gi in content if gi in match)
    lex = lex_hits / len(content) if content else 1.0

    rule_score = 0.0
    for gi, gt in enumerate(g):
        if gi not in match:
            continue
        pi, kind = match[gi]
        if kind == "exact":
            rule_score += 1.0
        elif gi in g2l:
            lemma = lemmas[g2l[gi]]
            pt = p[pi]
            if pt == lemma and gt != lemma:
                rule_score += 0.0  # rule not applied: rule-OFF form
            elif gt == lemma:
                rule_score += 0.0  # gold is bare lemma, pred altered it
            else:
                rule_score += 0.5  # attempted a change, wrong one
    rule = rule_score / len(g)

    seq_g = [gi for gi in range(len(g)) if gi in match]
    seq_p = sorted(seq_g, key=lambda gi: match[gi][0])
    order = lcs_len(seq_g, seq_p) / len(g) if seq_g else 0.0

    total = w["lex"] * lex + w["rule"] * rule + w["order"] * order + w["fmt"] * fmt
    # extra tokens the gold does not have (repetition, hedging) scale everything down
    extra = len(p) - len(used_p)
    if extra > 0:
        total *= max(0.0, 1 - extra / len(g))
    return {"lex": lex, "rule": rule, "order": order, "fmt": fmt, "total": total}


if __name__ == "__main__":
    gold = "aqègo açèqaqaqègo xadeoiçido"
    lemmas = ["açèqaqa", "xadeo"]
    for out in [
        "<translation>aqègo açèqaqaqègo xadeoiçido</translation>",
        "<translation>aqègo xadeoiçido açèqaqaqègo</translation>",
        "<translation>aqègo açèqaqa xadeoiçido</translation>",
        "<translation>aqègo açèqaqaqègo xadeo</translation>",
        "<translation>aqègo açèqaqaqègo xadeoiçido xadeoiçido xadeoiçido</translation>",
        "<translation>nothing here</translation>",
        "aqègo açèqaqaqègo xadeoiçido",
    ]:
        print(out[:60].ljust(62), {k: round(v, 2) for k, v in grade(out, gold, lemmas).items()})
