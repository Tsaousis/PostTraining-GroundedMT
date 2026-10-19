"""Hu-style CoT variant of the MIK train file: swap the no-CoT suffix for the CoT instruction (targets stay the gold translation)."""
import json, sys
from build_train import SUFFIX as OLD
COT = ("\n\nYou may explain your chain of thoughts prior to producing the required translation. IMPORTANT: Do write your "
       "translation between tags in the following manner: <translation>your translation here</translation>.")
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(dst, "w") as o:
    for l in open(src):
        r = json.loads(l); u = r["messages"][1]["content"]; assert u.endswith(OLD)
        r["messages"][1]["content"] = u[:-len(OLD)] + COT; o.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
print(n)
