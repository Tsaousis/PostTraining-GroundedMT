"""Re-score saved predictions with Marmonier et al.'s accuracy metric.

The eval dumps carry pred/gold in test-file order, so the partition label and the
adjective-order reference can be recovered by zipping against the source jsonl.
No GPU needed.
"""

import json
import os
import sys
from collections import defaultdict

TEST_DIR = "~/swells_min/SWELLS_Dataset/Test/Test_CoT"

PARTITIONS = ["1", "2m", "2f", "3", "4", "5m", "5f", "6", "7m", "7f", "8a", "8g", "9a", "9g"]


def norm(text):
    return (text or "").strip().strip("@").strip().lower().rstrip(".").strip()


def accuracy(pred, gold, adj_gold, partition):
    p, g = norm(pred), norm(gold)
    if not p:
        return 0.0
    if partition.startswith(("8", "9")):
        return 100.0 if sorted(p.split()) == sorted(g.split()) else 0.0
    if p == g:
        return 100.0
    if adj_gold and p == norm(adj_gold):
        return 50.0
    return 0.0


def source_rows(name):
    path = os.path.join(TEST_DIR, name + ".jsonl")
    ciphered = "eng_art" in name
    adj_key = (
        "ciphered_adjective_order_error_reference"
        if ciphered
        else "adjective_order_error_reference"
    )
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            gold = rec["ciphered_reference"] if ciphered else rec["reference"]
            if not (gold or "").strip().strip("@").strip():
                continue
            part = rec.get("instance_type", "").split(".")[0]
            rows.append((part if part not in ("6m", "6f") else "6",
                         (rec.get(adj_key) or "").strip().strip("@").strip()))
    return rows


def main(eval_dir, json_out=None):
    scores = {}
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".jsonl"):
            continue
        name = fn[:-6]
        preds = [json.loads(l) for l in open(os.path.join(eval_dir, fn))]
        rows = source_rows(name)[: len(preds)]
        if len(rows) != len(preds):
            print(f"{name}: length mismatch, skipped")
            continue
        per = defaultdict(list)
        for rec, (part, adj) in zip(preds, rows):
            per[part].append(accuracy(rec["pred"], rec["gold"], adj, part))
        allv = [v for vs in per.values() for v in vs]
        by_part = {p: sum(per[p]) / len(per[p]) for p in PARTITIONS if p in per}
        latin = [v for p, v in by_part.items() if p.startswith(("8", "9"))]
        scores[name] = {
            "n": len(allv),
            "average": sum(allv) / len(allv),
            "latin_8_9": sum(latin) / len(latin) if latin else None,
            "by_partition": by_part,
            "n_by_partition": {p: len(per[p]) for p in by_part},
        }
        print(f"\n{name}  (n={len(allv)}, mean acc {sum(allv)/len(allv):.1f})")
        for p, v in by_part.items():
            print(f"  {p:<3} {v:5.1f}  (n={len(per[p])})")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(scores, f, indent=2)
    return scores


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
