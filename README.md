# Lyrics AI Studio: генерация текстов песен через GUI

Теперь проект умеет **сам парсить тексты исполнителей** и автоматически добавлять JSON-теги (`meta`) для каждой песни.

## Что делает парсер автоматически

Для каждой песни формируется объект с:
- `messages` (для обучения модели),
- `meta.artist`,
- `meta.title`,
- `meta.source_file`,
- `meta.style_tags` (авто-теги по содержанию).

Пример одной строки в `parsed.jsonl`:

```json
{
  "messages": [...],
  "meta": {
    "artist": "artist_1",
    "title": "song_01",
    "source_file": "artist_1/song_01.txt",
    "style_tags": ["melancholic", "atmospheric"]
  }
}
```

---

## 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## 2) Как подготовить файлы исполнителей

Поддерживаются 2 формата.

### Вариант A (простой): 1 файл = 1 песня

```text
data/raw/
  artist_1/
    song_01.txt
    song_02.txt
  artist_2/
    song_01.txt
```

### Вариант B (автопарсинг нескольких песен из 1 файла)

В одном `.txt` можно хранить несколько песен, разделяя их строкой `---` или `===`:

```text
Artist: Artist One
Title: Night City
<текст песни>
---
Artist: Artist One
Title: Rain
<текст песни>
```

Поддерживаются заголовки:
- `Artist:` / `Исполнитель:`
- `Title:` / `Название:` / `Song:`

Если заголовков нет — артист берется из имени папки, название из имени файла.

---

## 3) Запуск GUI

```bash
python src/gui_app.py
```

Открой: `http://localhost:7860`

---

## 4) Что нажимать в GUI

### Вкладка 1: Подготовка датасета

Поля:
- `Папка с песнями`: `data/raw`
- `train`: `data/processed/train.jsonl`
- `valid`: `data/processed/valid.jsonl`
- `parsed`: `data/processed/parsed.jsonl` ← здесь будет полный распарсенный JSON с тегами

Нажми **Собрать датасет**.

### Вкладка 2: Обучение

Оставь дефолтные значения, либо подстрой под свою GPU.

### Вкладка 3: Генерация

Укажи тему/настроение/рифму и нажми **Сгенерировать**.

---

## 5) CLI (если нужно без GUI)

```bash
python src/prepare_dataset.py \
  --input_dir data/raw \
  --output_parsed data/processed/parsed.jsonl \
  --output_train data/processed/train.jsonl \
  --output_valid data/processed/valid.jsonl
```

```bash
python src/train_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/train.jsonl \
  --valid_file data/processed/valid.jsonl \
  --output_dir outputs/qwen-lyrics-lora \
  --use_4bit
```

```bash
python src/generate_lyrics.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen-lyrics-lora \
  --train_file data/processed/train.jsonl \
  --topic "ночной город" \
  --mood "меланхолично" \
  --rhyme "ABAB"
```
