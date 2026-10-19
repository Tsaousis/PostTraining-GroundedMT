"""chrF reward over the <translation> payload, plus the diagnostics the pilot needs.

The open question for GRPO on SWELLS is whether a ~37-character target leaves any
within-group spread once rewards are standardised: a group of rollouts that all
score identically contributes exactly zero gradient.  Every call is appended to
REWARD_LOG so `analyze_pilot.py` can report the zero-variance fraction.
"""

import json
import os
import re
import threading

import sacrebleu

TRANSLATION_RE = re.compile(r"<translation>(.*?)</translation>", re.S)
REWARD_LOG = os.environ.get("SWELLS_REWARD_LOG", "")
# SWELLS_STRICT_TAG=1: an unclosed <translation> tag scores 0 (no last-line fallback), so
# unfinished reasoning is never rewarded.
STRICT_TAG = os.environ.get("SWELLS_STRICT_TAG", "") == "1"
_lock = threading.Lock()


def extract_translation(solution_str):
    m = TRANSLATION_RE.findall(solution_str or "")
    if m:
        return m[-1].strip().strip("@").strip()
    if STRICT_TAG:
        return ""
    # Model never closed the tag: fall back to the last non-empty line so a
    # formatting slip does not read as a translation failure.
    lines = [l.strip() for l in (solution_str or "").splitlines() if l.strip()]
    return lines[-1].strip("@").strip() if lines else ""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    predicted = extract_translation(solution_str)
    if not predicted or not ground_truth:
        score = 0.0
    else:
        try:
            score = sacrebleu.sentence_chrf(predicted, [ground_truth]).score / 100.0
        except Exception:
            score = 0.0

    if REWARD_LOG:
        rec = {
            "score": score,
            "pred": predicted[:200],
            "gold": ground_truth[:200],
            "has_tag": bool(TRANSLATION_RE.search(solution_str or "")),
            "gen_chars": len(solution_str or ""),
        }
        if isinstance(extra_info, dict):
            rec["index"] = extra_info.get("index")
            rec["instance_type"] = extra_info.get("instance_type")
            rec["direction"] = extra_info.get("direction")
        with _lock:
            with open(REWARD_LOG, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return score
