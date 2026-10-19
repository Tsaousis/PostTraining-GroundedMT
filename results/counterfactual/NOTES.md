## Confound check (BAGS vs non-BAGS)
Only 38/600 items use a BAGS adjective (orig rule: adj-first; flipped: noun-first = anti-English). 562 non-BAGS (orig: noun-first; flipped: adj-first = English order).
SFT4B partial(400): BAGS edited 72% (n=25) vs ORIG prompts 39.5% noun-first -> shift also where flip is anti-English. nonBAGS edited 64.5%.
Base4B: no shift in either subset. Report confound honestly in paper; BAGS subset small.
Orig-prompt per-partition: SFT4B p5/p7 ~50/50, p2 60/40; 8B 85-92 (p2), 68-75 (p5/7).
## FULL RESULTS (eng->art, 600 adj-order items; rule-direction split 300/300)
Metric: % outputs adjective-first (English order) | rule says adj-first (Eng-like langs) / rule says noun-first
base4b  ORIG 30/36  CF 34/33   -> random, ignores rule
sft4b   ORIG 85/73  CF 83/80   -> English order regardless of rule (shortcut)
grpo4b nocot ORIG 61/51  CF 61/53 -> ignores rule, weaker Eng bias
grpo8b cot ORIG 72/14  -> tracks rule; CF pending
(engorder.py; CF flips rule so "keeps-orig" under CF = same order as ORIG)
