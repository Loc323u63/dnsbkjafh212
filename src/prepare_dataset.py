import argparse
import json
import random
import re
from pathlib import Path


SONG_SEPARATOR_PATTERNS = [
    re.compile(r"^\s*={3,}\s*$", re.MULTILINE),
    re.compile(r"^\s*-{3,}\s*$", re.MULTILINE),
]

HEADER_PATTERN = re.compile(
    r"^\s*(?:artist|исполнитель)\s*:\s*(?P<artist>.+?)\s*$\n?"
    r"^\s*(?:title|название|song)\s*:\s*(?P<title>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def infer_style_tags(lyrics: str):
    text = lyrics.lower()
    tags = []
    if any(w in text for w in ["любов", "сердц", "чувств", "слез"]):
        tags.append("romantic")
    if any(w in text for w in ["ноч", "тень", "дожд", "луна"]):
        tags.append("atmospheric")
    if any(w in text for w in ["улиц", "район", "бит", "дым"]):
        tags.append("urban")
    if any(w in text for w in ["боль", "пуст", "тоска", "одиноч"]):
        tags.append("melancholic")
    return tags[:4]


def build_prompt(artist: str, style_tags):
    style_line = ", ".join(style_tags) if style_tags else "без доп. тегов"
    return (
        "Ты помощник по написанию оригинальных текстов песен. "
        "Никогда не копируй дословно известные строки.\n"
        f"Стиль-ориентир: {artist}.\n"
        f"Теги стиля: {style_line}.\n"
        "Сгенерируй новый текст с образностью и рифмой."
    )


def parse_song_block(block: str, fallback_artist: str, source_file: str, fallback_title: str):
    block = block.strip()
    if not block:
        return None

    artist = fallback_artist
    title = fallback_title
    lyrics = block

    header = HEADER_PATTERN.search(block)
    if header:
        artist = header.group("artist").strip()
        title = header.group("title").strip()
        lyrics = block[header.end() :].strip()

    lyrics = normalize_text(lyrics)
    if len(lyrics) < 60:
        return None

    style_tags = infer_style_tags(lyrics)
    return {
        "messages": [
            {"role": "system", "content": "Ты пишешь только оригинальные тексты песен."},
            {"role": "user", "content": build_prompt(artist, style_tags)},
            {"role": "assistant", "content": lyrics},
        ],
        "meta": {
            "artist": artist,
            "title": title,
            "source_file": source_file,
            "style_tags": style_tags,
        },
    }


def split_songs_from_file(text: str):
    for pattern in SONG_SEPARATOR_PATTERNS:
        if pattern.search(text):
            return [part.strip() for part in pattern.split(text) if part.strip()]
    # fallback: one whole file = one song
    return [text.strip()] if text.strip() else []


def collect_examples(input_dir: Path):
    examples = []
    for artist_dir in sorted(input_dir.iterdir()):
        if not artist_dir.is_dir():
            continue
        fallback_artist = artist_dir.name
        for txt_file in sorted(artist_dir.glob("*.txt")):
            raw = txt_file.read_text(encoding="utf-8", errors="ignore")
            song_blocks = split_songs_from_file(raw)
            for idx, block in enumerate(song_blocks, start=1):
                fallback_title = txt_file.stem if len(song_blocks) == 1 else f"{txt_file.stem}_{idx}"
                ex = parse_song_block(
                    block=block,
                    fallback_artist=fallback_artist,
                    source_file=str(txt_file.relative_to(input_dir)),
                    fallback_title=fallback_title,
                )
                if ex:
                    examples.append(ex)
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
    parser.add_argument("--output_parsed", type=Path, default=None, help="Опционально: сохранить полный parsed JSONL до split")
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

    if args.output_parsed:
        write_jsonl(args.output_parsed, examples)

    random.seed(args.seed)
    random.shuffle(examples)

    split = int(len(examples) * (1 - args.valid_ratio))
    train_items = examples[:split]
    valid_items = examples[split:]

    write_jsonl(args.output_train, train_items)
    write_jsonl(args.output_valid, valid_items)

    print(f"Parsed songs: {len(examples)}")
    if args.output_parsed:
        print(f"Saved parsed: {args.output_parsed}")
    print(f"Saved train: {len(train_items)} -> {args.output_train}")
    print(f"Saved valid: {len(valid_items)} -> {args.output_valid}")


if __name__ == "__main__":
    main()
