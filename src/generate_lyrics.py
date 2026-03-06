import argparse
import json
from collections import Counter
from pathlib import Path


def ngrams(tokens, n=5):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def novelty_score(generated: str, train_texts, n=5) -> float:
    gen_tokens = generated.lower().split()
    gen_ngrams = Counter(ngrams(gen_tokens, n=n))
    if not gen_ngrams:
        return 1.0

    train_ngram_set = set()
    for text in train_texts:
        t = text.lower().split()
        train_ngram_set.update(ngrams(t, n=n))

    overlap = sum(count for gram, count in gen_ngrams.items() if gram in train_ngram_set)
    total = sum(gen_ngrams.values())
    return max(0.0, 1.0 - overlap / max(1, total))


def load_train_texts(train_file: Path):
    texts = []
    with train_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            assistant = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if assistant:
                texts.append(assistant[-1])
    return texts


def build_prompt(topic: str, mood: str, rhyme: str, verses: int) -> str:
    return (
        "Напиши полностью оригинальный текст песни на русском языке.\n"
        f"Тема: {topic}\n"
        f"Настроение: {mood}\n"
        f"Схема рифмы: {rhyme}\n"
        f"Количество куплетов: {verses}\n"
        "Требования:\n"
        "- Никаких дословных заимствований.\n"
        "- Яркие образы и метафоры.\n"
        "- Добавь припев после первого куплета."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--mood", type=str, default="лирично")
    parser.add_argument("--rhyme", type=str, default="ABAB")
    parser.add_argument("--verses", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    prompt = build_prompt(args.topic, args.mood, args.rhyme, args.verses)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    train_texts = load_train_texts(args.train_file)
    score = novelty_score(text, train_texts, n=5)

    print("=== GENERATED LYRICS ===")
    print(text)
    print("\n=== METRICS ===")
    print(f"novelty_score: {score:.3f}")


if __name__ == "__main__":
    main()
