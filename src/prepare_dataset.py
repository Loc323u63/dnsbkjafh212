import argparse
import json
import random
from pathlib import Path


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def build_prompt(artist: str) -> str:
    return (
        "Ты помощник по написанию оригинальных текстов песен. "
        "Никогда не копируй дословно известные строки.\n"
        f"Стиль-ориентир: {artist}.\n"
        "Сгенерируй новый текст с образностью и рифмой."
    )


def collect_examples(input_dir: Path):
    examples = []
    for artist_dir in sorted(input_dir.iterdir()):
        if not artist_dir.is_dir():
            continue
        artist = artist_dir.name
        for txt_file in sorted(artist_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8", errors="ignore")
            lyrics = normalize_text(raw)
            if len(lyrics) < 60:
                continue
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": "Ты пишешь только оригинальные тексты песен."},
                        {"role": "user", "content": build_prompt(artist)},
                        {"role": "assistant", "content": lyrics},
                    ]
                }
            )
    return examples


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_train", type=Path, required=True)
    parser.add_argument("--output_valid", type=Path, required=True)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Папка не найдена: {args.input_dir}")
    if not (0.01 <= args.valid_ratio <= 0.5):
        raise ValueError("valid_ratio должен быть в диапазоне [0.01, 0.5]")

    examples = collect_examples(args.input_dir)
    if len(examples) < 10:
        raise ValueError("Слишком мало примеров. Желательно хотя бы 10+ песен.")

    random.seed(args.seed)
    random.shuffle(examples)

    split = int(len(examples) * (1 - args.valid_ratio))
    train_items = examples[:split]
    valid_items = examples[split:]

    write_jsonl(args.output_train, train_items)
    write_jsonl(args.output_valid, valid_items)

    print(f"Saved train: {len(train_items)} -> {args.output_train}")
    print(f"Saved valid: {len(valid_items)} -> {args.output_valid}")


if __name__ == "__main__":
    main()
