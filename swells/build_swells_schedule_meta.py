"""Re-run build_swells_schedule with the same seeds and emit, per curriculum row,
the metadata grader v2 needs: the target gold string, the ciphered target-side
dictionary lemmas (the "rule-OFF" forms), direction and instance type.

Asserts the regenerated messages are byte-identical to the frozen curriculum so
the metadata is guaranteed to line up with the existing files.
"""
import argparse
import json
import re
from pathlib import Path

import build_swells_schedule as bss

TAG = re.compile(r"<translation>(.*?)</translation>", re.S)
_current = {}


def _meta_encode_row(item, path, modality, conlanger_cls, encode_no_rev, encode_rev):
    if modality == "cot":
        prompt = item["CoT_prompt_template"]
        completion = item["completion_template"]
    else:
        prompt = item["no_CoT_prompt_template"]
        completion = item["no_CoT_completion_template"]
    conlanger = conlanger_cls()
    encoder = encode_rev if path.parent.name.endswith("_rev") else encode_no_rev
    direction = "eng_art" if path.name.endswith("_eng_art.json") else "art_eng"
    if direction == "eng_art":
        # French_W: "@regarder@ (1st group v.): to watch |@genou@ (masc. n.): knee"
        lemmas = [m.group(0) for m in re.finditer(r"@[^@]+@", item["French_W"])]
        lemmas = [encoder(l, conlanger).strip() for l in lemmas]
    else:
        # English_W: "watch (v.) : @regarder@ (1st group v.) |knee : @genou@ (masc. n.)"
        lemmas = []
        for entry in item["English_W"].split("|"):
            head = entry.split(":")[0]
            head = re.sub(r"\(.*?\)", "", head).strip()
            if head:
                lemmas.append(head)
    row = {
        "messages": [
            {"role": "system", "content": "You are an expert linguist and translator."},
            {"role": "user", "content": encoder(prompt, conlanger)},
            {"role": "assistant", "content": encoder(completion, conlanger)},
        ]
    }
    gold = TAG.findall(row["messages"][2]["content"])[-1].strip().strip("@").strip()
    row["meta"] = {
        "direction": direction,
        "modality": modality,
        "instance_type": item["instance_type"],
        "rev": path.parent.name.endswith("_rev"),
        "gold": gold,
        "lemmas": lemmas,
        "french": item["French"].strip("@"),
        "english": item["English"],
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("~/swells_data/SWELLS_Dataset"))
    ap.add_argument("--repo", type=Path, default=Path("~/swells-official"))
    ap.add_argument("--frozen", type=Path, default=Path("${SWELLS_SCHEDULE:-data/swells_curriculum/sft}"))
    ap.add_argument("--output", type=Path, default=Path("${SWELLS_SCHEDULE:-data/swells_curriculum/sft}/meta"))
    args = ap.parse_args()

    bss.encode_row = _meta_encode_row
    tmp = args.output / "_regen"
    tmp.mkdir(parents=True, exist_ok=True)
    bss.build_courses(args.dataset, tmp, args.repo)
    bss.build_final(args.dataset, tmp, args.repo)

    for name in [f"course_{c}.jsonl" for c in bss.COURSE_COUNTS] + ["final_1408.jsonl"]:
        frozen = [json.loads(l) for l in (args.frozen / name).open()]
        regen = [json.loads(l) for l in (tmp / name).open()]
        assert len(frozen) == len(regen), name
        for a, b in zip(frozen, regen):
            assert a["messages"] == b["messages"], f"{name}: curriculum mismatch"
        (args.output / name).write_text(
            "".join(json.dumps(r["meta"], ensure_ascii=False) + "\n" for r in regen)
        )
        print("ok", name, len(regen))


if __name__ == "__main__":
    main()
