# AI-генератор текстов песен (оригинальные тексты в заданном стиле)

Готовый минимальный проект для:
- подготовки датасета из `.txt` файлов с текстами;
- дообучения базовой языковой модели через **LoRA**;
- генерации новых текстов по теме/настроению;
- базовой проверки на избыточное копирование из обучающих данных;
- запуска всего через простой **GUI**.

> ⚠️ Важно: использование чужих текстов может затрагивать авторские права. Используйте только лицензированные данные или собственные тексты.

## 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) GUI (самый простой способ)

Запуск:

```bash
python src/gui_app.py
```

Открой в браузере: `http://localhost:7860`

В GUI есть 3 вкладки:
1. **Подготовка датасета** (`data/raw -> data/processed/*.jsonl`)
2. **Обучение LoRA**
3. **Генерация текста** + `novelty_score`

## 3) Подготовка данных вручную

Положите файлы песен в структуру:

```text
data/raw/
  artist_a/
    song1.txt
    song2.txt
  artist_b/
    song1.txt
```

Каждый `.txt` — полный текст песни.

Соберите train/valid JSONL:

```bash
python src/prepare_dataset.py \
  --input_dir data/raw \
  --output_train data/processed/train.jsonl \
  --output_valid data/processed/valid.jsonl
```

## 4) Дообучение LoRA (CLI)

```bash
python src/train_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/train.jsonl \
  --valid_file data/processed/valid.jsonl \
  --output_dir outputs/qwen-lyrics-lora
```

## 5) Генерация текста (CLI)

```bash
python src/generate_lyrics.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen-lyrics-lora \
  --train_file data/processed/train.jsonl \
  --topic "ночной город, одиночество" \
  --mood "меланхолично, атмосферно" \
  --rhyme "ABAB" \
  --verses 2 \
  --max_new_tokens 280
```

Скрипт выводит:
- готовый текст;
- `novelty_score` (0..1): чем выше, тем меньше совпадений с train датасетом по n-граммам.

## 6) Что можно улучшить

- добавить reranking нескольких кандидатов по метрике новизны;
- добавить детектор похожести (MinHash/embeddings);
- добавить кнопку «сгенерировать 5 вариантов» и автосортировку по novelty;
- сделать экспорт в `.txt`/`.docx`.
