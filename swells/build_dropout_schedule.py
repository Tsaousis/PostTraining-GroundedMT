"""Annealing grammar-rule dropout curriculum for SWELLS (Zhu, Panigrahi & Arora 2025).

Regenerates the frozen curriculum (courses 1-7, then final_1408 x E epochs) from
the SWELLS templates with exactly the same items, order and conlangs, and
rewrites only the grammar-excerpt part of each prompt:

  needed rules  = the rule paragraphs of the item's own grammar excerpt
  distractors   = +25% rule paragraphs sampled from other partitions' excerpts
                  (same underlying language, encoded with the item's conlang)
  dropout       = every rule is dropped independently with
                  p(g) = min(1, g / (0.6 * N)), g = global example index over
                  the whole curriculum, N = total examples (courses + E finals)

Rules are the blank-line separated paragraphs of the excerpt; short heading
blocks are kept only while one of their rules survives. When no rule survives,
the grammar section is removed and the intro is rewritten to the
dictionary-only (W) prompt used by Test_CoT/*_dictionary_only.jsonl. Loss
targets (CoT / <translation>) are never changed, no item is dropped.

Training must consume each stage file sequentially (sft_train.py --no_shuffle,
grpo_train.py --no_shuffle); the within-stage order is already shuffled here.
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

import build_swells_schedule as base
from prepare_nocot import COT_INSTR, NOCOT_INSTR, TAG

GRAMMAR_RE = re.compile(
    r"\n\nAnd here are relevant excerpts from a grammar book: \n\n"
    r"Beginning of @French@ Grammar Book Excerpts\n\n"
    r"(?P<body>.*?)"
    r"\n\nEnd of @French@ Grammar Book Excerpts\n",
    re.S,
)
WITH_GRAMMAR = "with the help of a few dictionary entries and excerpts from a grammar book."
DICT_ONLY = "with the help of a few dictionary entries."


SEP_RE = re.compile(r"(\n[ \t]*\n(?:[ \t]*\n)*)")


def split_blocks(body):
    """[(block, following_separator)] preserving the original blank-line runs."""
    parts = SEP_RE.split(body)
    out = []
    for i in range(0, len(parts), 2):
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if parts[i].strip():
            out.append((parts[i], sep))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + parts[i] + sep)
    return out


def heading_level(block):
    """None for a rule paragraph, else 0 (section), 1 (subsection), 2 (minor/caption)."""
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if len(lines) > 3 or any(len(l) > 80 for l in lines):
        return None
    if lines[-1].endswith((".", "!", "?", ":", "@")):
        return None
    if len(lines) > 1 and not re.fullmatch(r"(.)\1{2,}", lines[0]):
        return None
    first = lines[0]
    if re.fullmatch(r"(.)\1{2,}", first) or first.isupper():
        return 0
    if first.startswith(("§§", "###", "-- ", "**", "++")):
        return 2
    if first.startswith(("§", "- ", "## ", "+ ")):
        return 1
    return 2


def parse_units(body):
    """Blocks -> units; a block ending in ':' is glued to the next (rule intro)."""
    blocks = split_blocks(body)
    units, i = [], 0
    while i < len(blocks):
        b, sep = blocks[i]
        lvl = heading_level(b)
        if lvl is None and b.rstrip().endswith(":") and i + 1 < len(blocks):
            b = b + sep + blocks[i + 1][0]
            sep = blocks[i + 1][1]
            i += 1
        units.append({"kind": "heading" if lvl is not None else "rule", "level": lvl, "text": b, "sep": sep})
        i += 1
    return units


def rule_set(template):
    m = GRAMMAR_RE.search(template)
    return [u["text"].strip() for u in parse_units(m.group("body")) if u["kind"] == "rule"]


def render(units):
    kept = []
    for i, u in enumerate(units):
        if u["kind"] == "rule":
            if u["keep"]:
                kept.append(u)
            continue
        for v in units[i + 1:]:
            if v["kind"] == "heading" and v["level"] <= u["level"]:
                break
            if v["kind"] == "rule" and v["keep"]:
                kept.append(u)
                break
    text = "".join(u["text"] + (u["sep"] or "\n\n") for u in kept)
    return text.rstrip("\n\t ")


def apply_dropout(template, p, pool, rng, distractor_frac):
    m = GRAMMAR_RE.search(template)
    assert m, "no grammar section in template"
    units = parse_units(m.group("body"))
    needed = [u for u in units if u["kind"] == "rule"]
    own = {u["text"] for u in needed}
    n_dis = int(math.ceil(distractor_frac * len(needed)))
    candidates = [r for r in pool if r not in own]
    distractors = rng.sample(candidates, n_dis)
    for d in distractors:
        pos = rng.randint(0, len(units))
        units.insert(pos, {"kind": "rule", "level": None, "text": d, "sep": "\n\n", "distractor": True})
    rules = [u for u in units if u["kind"] == "rule"]
    for u in rules:
        u["keep"] = rng.random() >= p
    info = {
        "p": round(p, 4),
        "n_needed": len(needed),
        "n_distractor": n_dis,
        "kept_needed": sum(u["keep"] for u in rules if not u.get("distractor")),
        "kept_distractor": sum(u["keep"] for u in rules if u.get("distractor")),
    }
    body = render(units)
    if body:
        body += m.group("body")[len(m.group("body").rstrip("\n\t ")):]
        new = template[: m.start()] + m.group(0).replace(m.group("body"), body) + template[m.end():]
    else:
        new = template[: m.start()] + template[m.end():]
        assert WITH_GRAMMAR in new
        new = new.replace(WITH_GRAMMAR, DICT_ONLY).replace(" text segment into English", " segment into English")
    return new, info


def collect_rows(dataset):
    """Same sampling + conlang RNG sequence as the frozen builder; returns stage -> rows."""
    conlanger_cls, enc_no_rev, enc_rev = base.load_conlanger(Path("~/swells-official"))
    data = dataset / "Train" / "Base"
    random.seed(0)
    sampling_rng = random.Random(0)
    stages = {}
    for course, total in base.COURSE_COUNTS.items():
        rows = []
        for (modality, direction), count in base.allocate(total).items():
            paths = base.course_paths(data, course, direction)
            for path, n in base.split_even(count, paths):
                for item in base.sample_items(path, n, sampling_rng):
                    rows.append((item, path, modality, conlanger_cls()))
        random.shuffle(rows)
        stages[f"course_{course}"] = rows
    sampling_rng = random.Random(1)
    rows = []
    for (modality, direction), count in base.allocate(1408).items():
        paths = []
        for course in base.COURSE_COUNTS:
            paths.extend(base.course_paths(data, course, direction))
        for path, n in base.split_even(count, paths):
            for item in base.sample_items(path, n, sampling_rng):
                rows.append((item, path, modality, conlanger_cls()))
    random.Random(1).shuffle(rows)
    stages["final_1408"] = rows
    return stages, enc_no_rev, enc_rev


def encode(text, path, conlanger, enc_no_rev, enc_rev):
    enc = enc_rev if path.parent.name.endswith("_rev") else enc_no_rev
    return enc(text, conlanger)


def to_nocot(messages):
    user = messages[1]["content"]
    assert COT_INSTR in user or NOCOT_INSTR in user
    gold = TAG.search(messages[2]["content"]).group(1).strip()
    return [
        messages[0],
        {"role": "user", "content": user.replace(COT_INSTR, NOCOT_INSTR)},
        {"role": "assistant", "content": f"<translation>{gold}</translation>"},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("~/swells_data/SWELLS_Dataset"))
    ap.add_argument("--frozen", type=Path, default=Path("${SWELLS_SCHEDULE:-data/swells_curriculum/sft}"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--final_epochs", type=int, required=True, help="4 for SFT, 2 for GRPO (matches the controls)")
    ap.add_argument("--anneal_frac", type=float, default=0.6)
    ap.add_argument("--distractor_frac", type=float, default=0.25)
    ap.add_argument("--prompt_token_budget", type=int, default=14336)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    args = ap.parse_args()

    stages, enc_no_rev, enc_rev = collect_rows(args.dataset)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Verify we reproduced the frozen curriculum exactly before touching anything.
    # The conlanger ciphers lazily (prompt first, then completion), so encode the frozen
    # pair in the original order and keep the completion; dropout prompts are ciphered
    # afterwards with the same (now fully populated) conlanger.
    for name, rows in stages.items():
        with (args.frozen / f"{name}.jsonl").open() as f:
            for i, ((item, path, modality, con), line) in enumerate(zip(rows, f, strict=True)):
                frozen = json.loads(line)["messages"]
                tmpl = item["CoT_prompt_template"] if modality == "cot" else item["no_CoT_prompt_template"]
                assert encode(tmpl, path, con, enc_no_rev, enc_rev) == frozen[1]["content"], (name, path)
                comp = item["completion_template"] if modality == "cot" else item["no_CoT_completion_template"]
                assert encode(comp, path, con, enc_no_rev, enc_rev) == frozen[2]["content"], (name, path)
                rows[i] = (item, path, modality, con, frozen[2]["content"])
        print(f"{name}: reproduced {len(rows)} rows (prompt + completion) byte-identically")

    pool = sorted({r for rows in stages.values() for item, *_ in rows for r in rule_set(item["CoT_prompt_template"])})
    print(f"distractor pool: {len(pool)} distinct rule paragraphs")

    order = [f"course_{c}" for c in base.COURSE_COUNTS] + ["final_1408"] * args.final_epochs
    N = sum(len(stages[s]) for s in order)
    T_anneal = args.anneal_frac * N
    print(f"N={N} examples, dropout reaches 100% at example {int(T_anneal)}")

    rng = random.Random(args.seed)
    out_cot, out_nocot, out_meta = args.output / "cot", args.output / "nocot", args.output / "meta"
    for d in (out_cot, out_nocot, out_meta):
        d.mkdir(parents=True, exist_ok=True)
    g = 0
    summary = []
    for stage_idx, name in enumerate(order):
        rows = list(stages[name])
        rng_stage = random.Random((args.seed, stage_idx))
        perm = list(range(len(rows)))
        rng_stage.shuffle(perm)
        fname = name if name.startswith("course") else f"final_ep{order[:stage_idx + 1].count('final_1408')}"
        fc, fn, fm = (open(d / f"{fname}.jsonl", "w") for d in (out_cot, out_nocot, out_meta))
        trimmed, lens, ps = 0, [], []
        for idx in perm:
            item, path, modality, con, target = rows[idx]
            p = min(1.0, g / T_anneal)
            tmpl = item["CoT_prompt_template"] if modality == "cot" else item["no_CoT_prompt_template"]
            budget_rng = random.Random((args.seed, stage_idx, idx))
            frac = args.distractor_frac
            while True:
                new, info = apply_dropout(tmpl, p, pool, random.Random(budget_rng.random()), frac)
                user = encode(new, path, con, enc_no_rev, enc_rev)
                n_tok = len(tok(user)["input_ids"])
                if n_tok <= args.prompt_token_budget or frac == 0:
                    break
                frac = 0.0
                trimmed += 1
            msgs = [
                {"role": "system", "content": "You are an expert linguist and translator."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": target},
            ]
            fc.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            fn.write(json.dumps({"messages": to_nocot(msgs)}, ensure_ascii=False) + "\n")
            info.update(global_index=g, stage=fname, frozen_index=idx, instance_type=item["instance_type"], prompt_tokens=n_tok)
            fm.write(json.dumps(info) + "\n")
            lens.append(n_tok)
            ps.append(p)
            g += 1
        for f in (fc, fn, fm):
            f.close()
        s = f"{fname}: {len(rows)} rows, p {ps[0]:.2f}->{ps[-1]:.2f}, prompt tokens max {max(lens)} mean {sum(lens) / len(lens):.0f}, distractors trimmed for budget: {trimmed}"
        print(s)
        summary.append(s)
    (args.output / "BUILD.txt").write_text(
        f"args={vars(args)}\nN={N}\npool={len(pool)}\n" + "\n".join(summary) + "\n"
    )


if __name__ == "__main__":
    main()
