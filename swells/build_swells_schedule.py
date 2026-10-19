import argparse
import json
import random
from pathlib import Path

import ijson


COURSE_COUNTS = {1: 276, 2: 1292, 3: 754, 4: 370, 5: 286, 6: 326, 7: 376}
WEIGHTS = {
    ("cot", "eng_art"): 100,
    ("cot", "art_eng"): 10,
    ("direct", "eng_art"): 10,
    ("direct", "art_eng"): 1,
}


def load_conlanger(repo: Path):
    notebook = json.loads((repo / "1_SWELLS_Training_Set_Generator.ipynb").read_text())
    namespace = {}
    for index in (5, 6, 7):
        source = "".join(notebook["cells"][index]["source"])
        source = source.replace("nltk.download('punkt_tab', quiet=True)", "")
        exec(source, namespace)
    return (
        namespace["Conlanger"],
        namespace["encode_at_strings_no_rev"],
        namespace["encode_at_strings_rev"],
    )


def allocate(total: int):
    denominator = sum(WEIGHTS.values())
    raw = {key: total * weight / denominator for key, weight in WEIGHTS.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    for key in sorted(raw, key=lambda item: raw[item] - result[item], reverse=True)[:remainder]:
        result[key] += 1
    return result


def split_even(total: int, paths):
    counts = [total // len(paths)] * len(paths)
    for index in range(total % len(paths)):
        counts[index] += 1
    return zip(paths, counts)


def sample_items(path: Path, count: int, rng):
    sample = []
    with path.open("rb") as handle:
        for index, item in enumerate(ijson.items(handle, "item")):
            if index < count:
                sample.append(item)
                continue
            replacement = rng.randint(0, index)
            if replacement < count:
                sample[replacement] = item
    if sample and len(sample) < count:
        source = list(sample)
        sample.extend(rng.choice(source) for _ in range(count - len(sample)))
    return sample


def course_paths(base: Path, course: int, direction: str):
    pattern = f"Tralalam_{course}*_train_*_{direction}.json"
    return sorted(base.glob(f"{direction}_*/*{pattern}"))


def encode_row(item, path, modality, conlanger_cls, encode_no_rev, encode_rev):
    if modality == "cot":
        prompt = item["CoT_prompt_template"]
        completion = item["completion_template"]
    else:
        prompt = item["no_CoT_prompt_template"]
        completion = item["no_CoT_completion_template"]

    conlanger = conlanger_cls()
    encoder = encode_rev if path.parent.name.endswith("_rev") else encode_no_rev
    return {
        "messages": [
            {"role": "system", "content": "You are an expert linguist and translator."},
            {"role": "user", "content": encoder(prompt, conlanger)},
            {"role": "assistant", "content": encoder(completion, conlanger)},
        ]
    }


def build_courses(dataset: Path, output: Path, repo: Path):
    conlanger_cls, encode_no_rev, encode_rev = load_conlanger(repo)
    base = dataset / "Train" / "Base"
    output.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    sampling_rng = random.Random(0)
    for course, total in COURSE_COUNTS.items():
        rows = []
        for (modality, direction), count in allocate(total).items():
            paths = course_paths(base, course, direction)
            if not paths:
                raise RuntimeError(f"No files for course {course}, {direction}")
            for path, path_count in split_even(count, paths):
                for item in sample_items(path, path_count, sampling_rng):
                    rows.append(
                        encode_row(
                            item,
                            path,
                            modality,
                            conlanger_cls,
                            encode_no_rev,
                            encode_rev,
                        )
                    )
        random.shuffle(rows)
        destination = output / f"course_{course}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        print(destination, len(rows))


def build_final(dataset: Path, output: Path, repo: Path):
    conlanger_cls, encode_no_rev, encode_rev = load_conlanger(repo)
    base = dataset / "Train" / "Base"
    sampling_rng = random.Random(1)
    rows = []
    for (modality, direction), count in allocate(1408).items():
        paths = []
        for course in COURSE_COUNTS:
            paths.extend(course_paths(base, course, direction))
        for path, path_count in split_even(count, paths):
            for item in sample_items(path, path_count, sampling_rng):
                rows.append(
                    encode_row(
                        item,
                        path,
                        modality,
                        conlanger_cls,
                        encode_no_rev,
                        encode_rev,
                    )
                )
    random.Random(1).shuffle(rows)
    destination = output / "final_1408.jsonl"
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    print(destination, len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path("~/swells_data/SWELLS_Dataset")
    )
    parser.add_argument(
        "--repo", type=Path, default=Path("~/swells-official")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("${SWELLS_SCHEDULE:-data/swells_curriculum/sft}")
    )
    args = parser.parse_args()
    build_courses(args.dataset, args.output, args.repo)
    build_final(args.dataset, args.output, args.repo)


if __name__ == "__main__":
    main()
