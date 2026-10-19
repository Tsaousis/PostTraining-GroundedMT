"""Answer the one question the pilot exists for: does chrF give GRPO a gradient?

GRPO standardises rewards inside each group of G rollouts, so a group whose
rollouts all score the same contributes nothing.  With ~37-character targets that
is a real risk, and it is cheaper to measure it than to discover it after a
multi-day run has flatlined.
"""

import argparse
import json
import statistics
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="./runs/pilot/rewards.jsonl")
    ap.add_argument("--group_size", type=int, default=8)
    args = ap.parse_args()

    recs = []
    with open(args.log) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not recs:
        print("no reward records")
        return

    scores = [r["score"] for r in recs]
    print(f"rollouts logged      : {len(recs)}")
    print(f"mean chrF            : {statistics.mean(scores):.4f}")
    print(f"median chrF          : {statistics.median(scores):.4f}")
    print(f"exact-zero rewards   : {sum(s == 0 for s in scores) / len(scores):.1%}")
    print(f"perfect (>=0.999)    : {sum(s >= 0.999 for s in scores) / len(scores):.1%}")
    print(f"emitted <translation>: {sum(bool(r.get('has_tag')) for r in recs) / len(recs):.1%}")

    # Group by prompt: consecutive rollouts of the same index form one GRPO group.
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        key = r.get("index")
        if key is None:
            key = i // args.group_size
        groups[key].append(r["score"])

    usable = [g for g in groups.values() if len(g) > 1]
    if not usable:
        print("\nnot enough rollouts per prompt to measure group variance")
        return

    stds = [statistics.pstdev(g) for g in usable]
    dead = sum(s < 1e-6 for s in stds)
    print(f"\ngroups                : {len(usable)} (mean {statistics.mean(len(g) for g in usable):.1f} rollouts)")
    print(f"zero-variance groups  : {dead / len(usable):.1%}  <-- contribute no gradient")
    print(f"mean within-group std : {statistics.mean(stds):.4f}")
    print(f"median within-group std: {statistics.median(stds):.4f}")

    by_type = defaultdict(list)
    for r in recs:
        by_type[(r.get("direction") or "?", (r.get("instance_type") or "?").split("__")[0])].append(r["score"])
    print("\nchrF by direction / phenomenon (top 12 by count):")
    for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {k[0]:8s} {k[1]:24s} n={len(v):5d} mean={statistics.mean(v):.3f}")

    verdict = "USABLE" if dead / len(usable) < 0.6 else "TOO SPARSE"
    print(f"\nverdict: {verdict} ({dead / len(usable):.0%} of groups are flat)")


if __name__ == "__main__":
    main()
