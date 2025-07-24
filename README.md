# PDF-Parser Project

Данный скрипт извлекает текст из PDF-документов с использованием OCR. Для улучшения корректности извлеченного текста используются модели LayoutParser и T5.

## Установка

-  Клонируем репозиторий и переходим в папку проекта (git clone <repository-url> & cd <repository-dir>)

- Создаем окружение (python -m venv venv или python3 -m venv venv)

- Активируем окружение (source venv/bin/activate)

- Устанавливаем зависимости (pip install -r requirements.txt)

- **Poppler**: Необходим для `pdf2image` для конвертации PDF в изображения.
  - **Ubuntu**: `sudo apt-get install poppler-utils`
  - **macOS**: `brew install poppler`
  - **Windows**: Загрузите и установите из [Poppler binaries](https://github.com/oschwartz10612/poppler-windows) и добавьте в PATH.
- **Tesseract**: Необходим для OCR. Установите его на вашу систему:
  - **Ubuntu**: `sudo apt-get install tesseract-ocr tesseract-ocr-rus`
  - **macOS**: `brew install tesseract tesseract-lang`
  - **Windows**: Загрузите и установите из [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) и добавьте в PATH.

- скачиваем данные для nlkt (python -c "import nltk; nltk.download('punkt')")

- Молодцы! Можно начинать работу.

## Функционал

- Запуск:

  Можно запускать скрипт с разными обработками, для этого добавлены опции
  python mayby_new_app.py [опция]

  - опция --disable_basic_normalization забускает скрипт без постобработки, текст обрабатывается лишь стандартными настройками модели LayoutParser.

  - без опции запускает скрипт со склейкой слов в блоках текста.

  - опция --with_smart_normalization включает дополнительную обработку текста моделью T5.

  - опция --study запускает обучение модели T5.

- Обучение модели T5:

  - для обучении модели сущуствует файл paraphrase_dataset.json, его нужно заполнить примерами приемлемых исправлений в некорректных предложениях.

## Составляющие кода

  - log добавляет логирование

  - normalize_text с помощью регулярок склеивает слова

  - clean_text удаляет лишнии символы, появление в тексте которых скорее всего вызвано некорректным извлечением.

  - NormalizationDataset подготавливает данные для обучения модели T5:
    - Переменные:
      - tokenizer - экземпляр T5Tokenizer
      - data_path - путь к json
      - max_length - максимальная длина последоватльности токенов
    -
