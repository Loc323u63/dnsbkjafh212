# Lyrics AI Studio: тексты из интернета → txt → парсинг → обучение → генерация

Теперь можно полностью по шагам:
1. В GUI ввести артистов.
2. Автоматически скачать тексты из интернета в `data/raw/<artist>/*.txt`.
3. Автоматически распарсить и разметить JSON-тегами.
4. Дообучить LoRA и генерировать новые тексты.

> Важно: соблюдай авторские права и условия источников текстов.

## 1) Установка

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Если в PowerShell появляется ошибка про запрет запуска скриптов, выполни один раз:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -U pip
pip install -r requirements.txt
```

## 2) Запуск GUI

```bash
python src/gui_app.py
```

Открой: `http://localhost:7860`

## 3) Полный workflow в GUI

### Вкладка 0: Скачать тексты из интернета
- `Исполнители`: выпадающий список с поиском по мере ввода (можно выбрать несколько и добавить свои)
- `Источники поиска`: выпадающий список с мультивыбором (`itunes`, `youtube`, `soundcloud`, `vk`)
- `Куда сохранить txt`: `data/raw`
- `Песен на исполнителя`: обычно 10–30
- `Пауза между запросами`: 0.25

Нажми **Скачать тексты**.

### Вкладка 1: Подготовка датасета
- `data/raw` -> `data/processed/parsed.jsonl`, `train.jsonl`, `valid.jsonl`
- Нажми **Собрать датасет**.

`parsed.jsonl` содержит `meta`:
- `artist`
- `title`
- `source_file`
- `style_tags`

### Вкладка 2: Обучение LoRA
- Укажи модель и пути
- Нажми **Запустить обучение**

### Вкладка 3: Генерация
- Укажи тему/настроение/рифму
- Нажми **Сгенерировать**

## 4) Как работает автозагрузка

Скрипт `src/fetch_lyrics.py`:
- ищет треки артиста через источники: `itunes`, `youtube`, `soundcloud`, `vk`;
- пробует получить текст по каждой песне через `lyrics.ovh`, затем fallback в `lrclib`;
- сохраняет найденные тексты в `.txt`.

CLI пример:

```bash
python src/fetch_lyrics.py \
  --artists "Miyagi, Скриптонит" \
  --sources "itunes,youtube,soundcloud,vk" \
  --output_dir data/raw \
  --max_songs 20
```

## 5) Багчекинг и стабильность

Что уже учтено:
- проверка пустого списка артистов;
- обработка сетевых ошибок по источникам;
- фильтрация слишком коротких текстов;
- безопасные имена файлов;
- fallback-источник для текста (`lrclib`).
