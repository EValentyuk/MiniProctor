# Заливка MiniProctor на GitHub

Краткая инструкция на потом. Профиль: https://github.com/EValentyuk.

## Перед заливкой -- проверить

- В `.gitignore` есть `data/`, `.venv/`, `*.pt`, `__pycache__/`, `.ipynb_checkpoints/`. Это уже сделано.
- В репозитории не должно быть личных видео с вебки (`data/raw/smoke.mp4` и др.) -- они под `.gitignore`, но проверить отдельно.
- Веса моделей `models/*.pt` и `*.task` не коммитим, они большие и качаются автоматически. Добавить в `.gitignore` папку `models/` или паттерны.
- В коде нет токенов, паролей, путей вида `c:\Users\ВЕГ\...`.

## Быстрый путь через `gh` CLI (рекомендую)

`gh` уже установлен и авторизован под аккаунтом `EValentyuk` (проверено 2026-04-15, scopes `repo, gist, read:org`). Репозитория `MiniProctor` нет, имя свободно. Поэтому весь процесс -- четыре команды без захода в браузер:

```bash
cd c:/Projects/MiniProctor
git init
git add .
git commit -m "Инициализация: бриф, smoke-тесты MediaPipe и YOLOv8"
gh repo create MiniProctor --public --source=. --remote=origin --push \
  --description "MVP-система прокторинга на pretrained CV-моделях. Pet-проект для портфолио"
```

После этого репозиторий доступен на https://github.com/EValentyuk/MiniProctor. Темы и README на английском добавляются отдельно (см. раздел «Что добавить на странице репозитория»).

Перед `git add .` всё равно проверить `git status` -- что в индекс не попадают `data/`, `.venv/`, `models/`. Если попали -- доправить `.gitignore` и `git rm --cached <файл>`.

## Ручной путь через веб-интерфейс

Этот путь оставлен на случай, если `gh` CLI вдруг не сработает.

### 1. Создать репозиторий на GitHub

- Зайти на https://github.com/new.
- Owner -- EValentyuk, имя -- `MiniProctor`.
- Описание: «MVP-система прокторинга на pretrained CV-моделях. Pet-проект для портфолио».
- Public.
- НЕ инициализировать README, .gitignore и license через GitHub -- они уже есть локально.

### 2. Локально инициализировать git и сделать первый коммит

В терминале из корня проекта:

```bash
cd c:/Projects/MiniProctor
git init
git add .gitignore CLAUDE.md MEMORY.md README.md MiniProctor-results.md requirements.txt docs/ src/ notebooks/
git status
```

Проверить, что в `git status` нет ничего из `data/`, `.venv/`, `models/`. Если есть -- доправить `.gitignore`.

```bash
git commit -m "Инициализация: бриф, структура, smoke-тесты MediaPipe и YOLOv8"
```

### 3. Подключить удалённый репозиторий и запушить

```bash
git branch -M main
git remote add origin https://github.com/EValentyuk/MiniProctor.git
git push -u origin main
```

При первом пуше GitHub попросит логин или Personal Access Token. Token создаётся в Settings -> Developer settings -> Personal access tokens -> Tokens (classic), scope `repo`.

### 4. Что добавить на странице репозитория после пуша

- **About** справа: короткое описание, темы (`computer-vision`, `mediapipe`, `yolov8`, `proctoring`, `python`).
- **Releases** -- если веса модели после fine-tune захочется выложить, сделать через GitHub Releases, не через основной репозиторий (там лимит 100 МБ на файл).
- **README на английском** -- добавить как `README.en.md` или сделать второй раздел в основном README.

## Что НЕ заливать никогда

- Личные видео экзаменуемых -- даже свои, для портфолио лучше нарезать только короткие демо без лица крупным планом.
- Веса fine-tuned моделей в основной коммит. Только в Releases или вообще на HuggingFace Hub.
- Токены GitHub, Roboflow API key и любые другие ключи. Если они когда-то попадут в код -- использовать `.env` и `python-dotenv`, а `.env` в `.gitignore`.

## Если что-то залилось ошибочно

- Файл попал в коммит -- удалить через `git rm --cached <файл>`, добавить в `.gitignore`, закоммитить.
- Файл попал в историю много коммитов назад -- понадобится `git filter-repo` или BFG Repo-Cleaner. Это уже не быстрая операция, лучше предотвратить.
- Если запушен токен -- немедленно отозвать токен в GitHub Settings, потом чистить историю.
