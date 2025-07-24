import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from PIL import Image

INPUT_PDF = "9dHREWMvdDMIM-страницы-1-страницы-2.pdf"
OUTPUT_DIR = "parsed_book"
DPI = 500
COLUMN_MARGIN = 100

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def normalize_text(text):
    """Улучшенная обработка текста с исправлением переносов"""
    # Склеивание переносов типа "ориги- \n нале"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Удаление одиноких дефисов в конце строк
    text = re.sub(r'-\s*\n', '', text)

    # Склеивание разорванных фраз между колонками
    text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', text)

    # Удаление лишних пробелов и переносов
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def detect_columns(image):
    """Автоматическое определение колонок на странице"""
    img = np.array(image.convert('L'))
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Вертикальная проекция черных пикселей
    vertical_projection = np.sum(binary, axis=0)

    # Поиск пробелов между колонками
    in_column = False
    columns = []
    start_idx = 0

    for idx, value in enumerate(vertical_projection):
        if value > 0 and not in_column:
            in_column = True
            start_idx = idx
        elif value == 0 and in_column:
            # Проверяем, достаточно ли широк пробел для колонки
            if idx - start_idx > COLUMN_MARGIN:
                in_column = False
                columns.append((start_idx, idx))

    # Если нашли меньше 2 колонок, используем дефолтное разделение
    if len(columns) < 2:
        width = image.width
        return [
            (0, width//2 - COLUMN_MARGIN//2),
            (width//2 + COLUMN_MARGIN//2, width)
        ]

    return columns

def process_columns(image, columns):
    """Обработка каждой колонки отдельно"""
    column_texts = []

    for i, (start, end) in enumerate(columns):
        # Вырезаем колонку с небольшим отступом
        column_img = image.crop((max(0, start-10), 0, min(image.width, end+10), image.height))

        # Распознавание с улучшенными параметрами
        text = pytesseract.image_to_string(
            column_img,
            lang='rus+eng',
            config=(
                '--oem 1 '
                '--psm 6 '
                '-c preserve_interword_spaces=1 '
                '-c textord_tabfind_find_tables=0 '  # Игнорируем таблицы
            )
        )
        column_texts.append(text)

    # Объединяем колонки в правильном порядке
    return "\n\n".join(column_texts)

def preprocess_image(image):
    """Улучшение качества изображения с акцентом на текст"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Увеличение контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Адаптивная бинаризация
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    # Удаление шумов
    denoised = cv2.fastNlMeansDenoising(
        thresh, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    return Image.fromarray(denoised)

def process_book():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Начало обработки: {INPUT_PDF}")

    try:
        pages = convert_from_path(INPUT_PDF, dpi=DPI)
    except Exception as e:
        log(f"Ошибка конвертации: {str(e)}")
        return

    full_text = ""
    for i, page in enumerate(pages, 1):
        try:
            log(f"Обработка страницы {i}/{len(pages)}")

            # Предобработка изображения
            processed_page = preprocess_image(page)

            # Определение колонок
            columns = detect_columns(processed_page)
            # Обработка колонок
            raw_text = process_columns(processed_page, columns)

            # Постобработка текста
            clean_text = normalize_text(raw_text)
            full_text += f"--- СТРАНИЦА {i} ---\n{clean_text}\n\n"

            # Сохранение постранично
            with open(os.path.join(OUTPUT_DIR, f"page_{i:03d}.txt"), 'w', encoding='utf-8') as f:
                f.write(clean_text)

        except Exception as e:
            log(f"Ошибка на странице {i}: {str(e)}")

    # Сохранение полного текста
    with open(os.path.join(OUTPUT_DIR, "full_book.txt"), 'w', encoding='utf-8') as f:
        f.write(full_text)

    log(f"Обработка завершена! Результаты в папке: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_book()
