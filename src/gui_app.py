import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import gradio as gr

from generate_lyrics import build_prompt, load_train_texts, novelty_score
from prepare_dataset import collect_examples, write_jsonl

ROOT_DIR = Path(__file__).resolve().parent.parent

POPULAR_ARTISTS = [
    "Скриптонит",
    "Miyagi",
    "Oxxxymiron",
    "Noize MC",
    "MORGENSHTERN",
    "Eminem",
    "The Weeknd",
    "Drake",
    "Kendrick Lamar",
    "Travis Scott",
]

SOURCE_CHOICES = ["itunes", "youtube", "soundcloud", "vk"]


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def _join_values(values) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        return values
    if isinstance(values, Iterable):
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return ",".join(cleaned)
    return str(values)


def run_fetch(artists, sources, output_dir: str, max_songs: int, sleep_s: float):
    artists_csv = _join_values(artists)
    sources_csv = _join_values(sources)

    if not artists_csv.strip():
        return "❌ Укажи хотя бы одного исполнителя."

    out_path = _resolve_path(output_dir)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "fetch_lyrics.py"),
        "--artists",
        artists_csv,
        "--sources",
        sources_csv,
        "--output_dir",
        str(out_path),
        "--max_songs",
        str(int(max_songs)),
        "--sleep_s",
        str(float(sleep_s)),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT_DIR))
    except Exception as exc:
        return f"❌ Ошибка запуска загрузчика: {exc}"

    if proc.returncode != 0:
        return f"❌ Загрузка завершилась с ошибкой\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"

    return f"✅ Тексты загружены в {out_path}\n\n{proc.stdout[-4000:]}"


def run_prepare(input_dir: str, output_train: str, output_valid: str, output_parsed: str, valid_ratio: float, seed: int):
    import random

    in_dir = _resolve_path(input_dir)
    train_path = _resolve_path(output_train)
    valid_path = _resolve_path(output_valid)

    if not in_dir.exists():
        return f"❌ Папка не найдена: {in_dir}"
    if not (0.01 <= float(valid_ratio) <= 0.5):
        return "❌ valid_ratio должен быть в диапазоне [0.01, 0.5]"

    examples = collect_examples(in_dir)
    if len(examples) < 10:
        return "❌ Слишком мало данных. Добавь минимум 10+ песен в data/raw/<artist>/*.txt"

    random.seed(int(seed))
    random.shuffle(examples)
    split = int(len(examples) * (1 - float(valid_ratio)))
    train_items = examples[:split]
    valid_items = examples[split:]

    parsed_path = _resolve_path(output_parsed)
    write_jsonl(parsed_path, examples)
    write_jsonl(train_path, train_items)
    write_jsonl(valid_path, valid_items)

    return (
        "✅ Датасет собран и размечен JSON-тегами\n"
        f"parsed: {len(examples)} -> {parsed_path}\n"
        f"train: {len(train_items)} -> {train_path}\n"
        f"valid: {len(valid_items)} -> {valid_path}"
    )


def run_training(base_model: str, train_file: str, valid_file: str, output_dir: str, max_length: int, batch_size: int, epochs: int, lr: float, use_4bit: bool):
    train_path = _resolve_path(train_file)
    valid_path = _resolve_path(valid_file)
    output_path = _resolve_path(output_dir)

    if not train_path.exists() or not valid_path.exists():
        return "❌ Не найдены train/valid файлы. Сначала собери датасет на вкладке 1."

    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "train_lora.py"),
        "--base_model",
        base_model,
        "--train_file",
        str(train_path),
        "--valid_file",
        str(valid_path),
        "--output_dir",
        str(output_path),
        "--max_length",
        str(max_length),
        "--batch_size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
    ]
    if use_4bit:
        cmd.append("--use_4bit")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=str(ROOT_DIR))
    except Exception as exc:
        return f"❌ Ошибка запуска обучения: {exc}"

    if proc.returncode != 0:
        return f"❌ Обучение завершилось с ошибкой\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"

    _load_model.cache_clear()
    return f"✅ Обучение завершено\n\n{proc.stdout[-4000:]}"


@lru_cache(maxsize=2)
def _load_model(base_model: str, adapter_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def run_generate(base_model: str, adapter_dir: str, train_file: str, topic: str, mood: str, rhyme: str, verses: int, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float):
    import torch

    adapter_path = _resolve_path(adapter_dir)
    train_path = _resolve_path(train_file)

    if not adapter_path.exists():
        return "", "❌ Папка адаптера не найдена. Сначала запусти обучение."
    if not train_path.exists():
        return "", "❌ Train file не найден для novelty_score."

    prompt = build_prompt(topic, mood, rhyme, verses)

    try:
        tokenizer, model = _load_model(base_model, str(adapter_path))
    except Exception as exc:
        return "", f"❌ Ошибка загрузки модели: {exc}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    score = novelty_score(text, load_train_texts(train_path), n=5)

    metrics = {
        "novelty_score": round(score, 3),
        "topic": topic,
        "mood": mood,
        "rhyme": rhyme,
    }
    return text, json.dumps(metrics, ensure_ascii=False, indent=2)


def build_app():
    with gr.Blocks(title="Lyrics AI Studio") as demo:
        gr.Markdown(
            "# 🎵 Lyrics AI Studio\n"
            "Пошаговый GUI: скачать тексты → подготовить датасет → обучить LoRA → сгенерировать текст."
        )

        with gr.Tab("0) Скачать тексты из интернета"):
            artists = gr.Dropdown(
                choices=POPULAR_ARTISTS,
                value=["Miyagi", "Скриптонит"],
                multiselect=True,
                allow_custom_value=True,
                filterable=True,
                label="Исполнители (можно печатать и выбирать)",
            )
            sources = gr.Dropdown(
                choices=SOURCE_CHOICES,
                value=SOURCE_CHOICES,
                multiselect=True,
                allow_custom_value=False,
                filterable=True,
                label="Источники поиска",
            )
            fetch_output_dir = gr.Textbox(value="data/raw", label="Куда сохранить txt")
            max_songs = gr.Slider(5, 50, value=20, step=1, label="Песен на исполнителя")
            sleep_s = gr.Slider(0.0, 2.0, value=0.25, step=0.05, label="Пауза между запросами")
            fetch_btn = gr.Button("Скачать тексты")
            fetch_out = gr.Textbox(label="Лог загрузки", lines=10)
            fetch_btn.click(run_fetch, [artists, sources, fetch_output_dir, max_songs, sleep_s], fetch_out)

        with gr.Tab("1) Подготовка датасета"):
            input_dir = gr.Textbox(value="data/raw", label="Папка с песнями (data/raw/<artist>/*.txt)")
            output_train = gr.Textbox(value="data/processed/train.jsonl", label="Куда сохранить train")
            output_valid = gr.Textbox(value="data/processed/valid.jsonl", label="Куда сохранить valid")
            output_parsed = gr.Textbox(value="data/processed/parsed.jsonl", label="Куда сохранить parsed с JSON-тегами")
            valid_ratio = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Доля valid")
            seed = gr.Number(value=42, precision=0, label="Seed")
            prepare_btn = gr.Button("Собрать датасет")
            prepare_out = gr.Textbox(label="Лог", lines=8)
            prepare_btn.click(run_prepare, [input_dir, output_train, output_valid, output_parsed, valid_ratio, seed], prepare_out)

        with gr.Tab("2) Обучение LoRA"):
            base_model = gr.Textbox(value="Qwen/Qwen2.5-1.5B-Instruct", label="Base model")
            train_file = gr.Textbox(value="data/processed/train.jsonl", label="Train file")
            valid_file = gr.Textbox(value="data/processed/valid.jsonl", label="Valid file")
            output_dir = gr.Textbox(value="outputs/qwen-lyrics-lora", label="Папка адаптера")
            max_length = gr.Slider(256, 2048, value=1024, step=128, label="Max length")
            batch_size = gr.Slider(1, 8, value=2, step=1, label="Batch size")
            epochs = gr.Slider(1, 10, value=2, step=1, label="Epochs")
            lr = gr.Number(value=2e-4, label="Learning rate")
            use_4bit = gr.Checkbox(value=True, label="Использовать 4-bit (только CUDA)")
            train_btn = gr.Button("Запустить обучение")
            train_out = gr.Textbox(label="Лог обучения", lines=14)
            train_btn.click(
                run_training,
                [base_model, train_file, valid_file, output_dir, max_length, batch_size, epochs, lr, use_4bit],
                train_out,
            )

        with gr.Tab("3) Генерация"):
            g_base_model = gr.Textbox(value="Qwen/Qwen2.5-1.5B-Instruct", label="Base model")
            g_adapter_dir = gr.Textbox(value="outputs/qwen-lyrics-lora", label="Папка адаптера")
            g_train_file = gr.Textbox(value="data/processed/train.jsonl", label="Train file (для novelty score)")
            topic = gr.Textbox(value="ночной город, одиночество", label="Тема")
            mood = gr.Textbox(value="меланхолично, атмосферно", label="Настроение")
            rhyme = gr.Textbox(value="ABAB", label="Схема рифмы")
            verses = gr.Slider(1, 4, value=2, step=1, label="Куплетов")
            max_new_tokens = gr.Slider(64, 600, value=280, step=8, label="Max new tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p")
            repetition_penalty = gr.Slider(1.0, 1.4, value=1.15, step=0.01, label="Repetition penalty")
            gen_btn = gr.Button("Сгенерировать")
            lyrics_out = gr.Textbox(label="Сгенерированный текст", lines=14)
            metrics_out = gr.Code(label="Метрики", language="json")

            gen_btn.click(
                run_generate,
                [g_base_model, g_adapter_dir, g_train_file, topic, mood, rhyme, verses, max_new_tokens, temperature, top_p, repetition_penalty],
                [lyrics_out, metrics_out],
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
