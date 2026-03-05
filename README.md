# Lyrics AI Studio: генерация текстов песен через GUI

Этот проект позволяет без программирования:
1. Подготовить датасет из `.txt` файлов песен.
2. Дообучить модель в стиле твоего набора данных (LoRA).
3. Генерировать новые тексты через браузер.

> Важно: используй только лицензированные тексты или свои собственные.

---

## Что нужно перед стартом

- Установленный **Python 3.10+**.
- Интернет (для загрузки базовой модели).
- Желательно GPU (CUDA). На CPU тоже возможно, но медленно.

Проверка:

```bash
python --version
```

---

## Шаг 1. Скачать и установить зависимости

Открой терминал в папке проекта и выполни:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Если ты на Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

---

## Шаг 2. Подготовить тексты (куда и как класть файлы)

Сделай структуру папок:

```text
data/raw/
  artist_1/
    song_01.txt
    song_02.txt
  artist_2/
    song_01.txt
```

Правила:
- 1 файл = 1 песня.
- Формат UTF-8.
- Рекомендую минимум 10–30 песен, лучше больше.

---

## Шаг 3. Запуск GUI в браузере

В терминале из корня проекта:

```bash
python src/gui_app.py
```

После запуска открой в браузере:

- `http://localhost:7860`

Если открываешь на другой машине в локальной сети, используй IP компьютера, где запущен скрипт, например `http://192.168.1.10:7860`.

---

## Шаг 4. Что нажимать в GUI (полная инструкция)

### Вкладка 1: «Подготовка датасета»

Поля:
- `Папка с песнями` → `data/raw`
- `Куда сохранить train` → `data/processed/train.jsonl`
- `Куда сохранить valid` → `data/processed/valid.jsonl`
- `Доля valid` → обычно `0.1`
- `Seed` → `42`

Нажми **«Собрать датасет»**.

Ожидаемый результат в логе:
- `✅ Датасет собран`
- числа по train/valid.

### Вкладка 2: «Обучение LoRA»

Минимальные значения:
- `Base model`: `Qwen/Qwen2.5-1.5B-Instruct`
- `Train file`: `data/processed/train.jsonl`
- `Valid file`: `data/processed/valid.jsonl`
- `Папка адаптера`: `outputs/qwen-lyrics-lora`
- `Max length`: `1024`
- `Batch size`: `1-2` (для слабой GPU)
- `Epochs`: `2-4`
- `Learning rate`: `2e-4`
- `Использовать 4-bit`: включай, если есть CUDA GPU

Нажми **«Запустить обучение»** и дождись `✅ Обучение завершено`.

### Вкладка 3: «Генерация»

Заполни:
- `Base model`: как на обучении
- `Папка адаптера`: `outputs/qwen-lyrics-lora`
- `Train file`: `data/processed/train.jsonl`
- `Тема`, `Настроение`, `Схема рифмы`, `Куплетов`

Нажми **«Сгенерировать»**.

Получишь:
- текст песни,
- метрику `novelty_score` (чем выше, тем меньше пересечений с train по n-граммам).

---

## Частые проблемы и что делать

### 1) Браузер не открывается
- Проверь, что в терминале есть строка с `Running on ...:7860`.
- Проверь порт:
  ```bash
  ss -ltnp | rg 7860
  ```
- Убедись, что запускаешь `python src/gui_app.py` из корня проекта.

### 2) Ошибка про `torch` / `transformers`
Повтори установку зависимостей в активированном `.venv`.

### 3) Обучение падает из-за памяти GPU
- Уменьши `Batch size` до 1.
- Уменьши `Max length` до 512.
- Оставь включенным `4-bit`.

### 4) На CPU слишком медленно
Это нормально. Для комфортной работы лучше CUDA GPU.

---

## CLI команды (если захочешь без GUI)

Подготовка:

```bash
python src/prepare_dataset.py \
  --input_dir data/raw \
  --output_train data/processed/train.jsonl \
  --output_valid data/processed/valid.jsonl
```

Обучение:

```bash
python src/train_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/train.jsonl \
  --valid_file data/processed/valid.jsonl \
  --output_dir outputs/qwen-lyrics-lora \
  --use_4bit
```

Генерация:

```bash
python src/generate_lyrics.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir outputs/qwen-lyrics-lora \
  --train_file data/processed/train.jsonl \
  --topic "ночной город, одиночество" \
  --mood "меланхолично, атмосферно" \
  --rhyme "ABAB" \
  --verses 2 \
  --max_new_tokens 280 \
  --temperature 0.9 \
  --top_p 0.95 \
  --repetition_penalty 1.15
```
