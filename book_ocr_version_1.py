import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from PIL import Image, ImageFilter, ImageDraw
import requests
from io import BytesIO
from spellchecker import SpellChecker

# ===== КОНФИГУРАЦИЯ =====
INPUT_PDF = "9dHREWMvdDMIM-страницы-1.pdf"
OUTPUT_DIR = "parsed_book"
DPI = 500
COLUMN_MARGIN = 70
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
USE_SPELLCHECK = True  # Включить/выключить проверку орфографии
# ========================

# Инициализация спеллчекера для русского и английского
spell_ru = SpellChecker(language='ru')
spell_en = SpellChecker(language='en')


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def correct_spelling(text):
    """Коррекция орфографии с использованием спеллчекера и словаря"""
    if not USE_SPELLCHECK:
        return text

    # Сначала применяем словарь частых ошибок
    for wrong, correct in CORRECTION_DICT.items():
        text = re.sub(r'\b' + wrong + r'\b', correct, text, flags=re.IGNORECASE)

    # Разбиваем текст на слова и исправляем
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    corrected_words = []

    for word in words:
        # Пропускаем пунктуацию и цифры
        if re.match(r'^[^\w\s]$', word) or word.isdigit():
            corrected_words.append(word)
            continue

        # Определяем язык слова
        has_cyrillic = bool(re.search('[а-яА-Я]', word))
        spell = spell_ru if has_cyrillic else spell_en

        # Если слово в словаре или слишком короткое - пропускаем
        if len(word) < 3 or word in spell:
            corrected_words.append(word)
            continue

        # Получаем наиболее вероятное исправление
        correction = spell.correction(word)

        # Если исправление отличается от оригинала и не None
        if correction and correction != word:
            # Сохраняем регистр оригинала
            if word.istitle():
                correction = correction.title()
            elif word.isupper():
                correction = correction.upper()

            corrected_words.append(correction)
            log(f"Исправлено: {word} → {correction}")
        else:
            corrected_words.append(word)

    return ''.join(corrected_words)

def normalize_text(text):
    """Улучшенная обработка текста с исправлением переносов и специальных символов"""
    # Склеивание переносов типа "ориги- \n нале"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # Удаление одиноких дефисов в конце строк
    text = re.sub(r'-\s*\n', '', text)

    # Склеивание разорванных фраз между колонками
    text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1\2', text)

    # Удаление лишних пробелов и переносов
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Коррекция орфографии
    text = correct_spelling(text)

    return text.strip()

def detect_columns(image):
    """Улучшенное определение колонок с анализом гистограммы"""
    # Конвертация в grayscale
    img = np.array(image.convert('L'))

    # Адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10
    )

    # Вертикальная проекция
    vertical_projection = np.sum(binary, axis=0) / 255

    # Находим пробелы между колонками
    gap_threshold = np.mean(vertical_projection) * 0.3
    in_gap = False
    gaps = []
    start_idx = 0

    for idx, value in enumerate(vertical_projection):
        if value < gap_threshold and not in_gap:
            in_gap = True
            start_idx = idx
        elif value >= gap_threshold and in_gap:
            in_gap = False
            if idx - start_idx > COLUMN_MARGIN:
                gaps.append((start_idx, idx))

    # Если нашли пробелы, определяем колонки
    if gaps:
        columns = []
        last_end = 0

        for start, end in gaps:
            if start - last_end > 100:  # Минимальная ширина колонки
                columns.append((last_end, start))
            last_end = end

        if image.width - last_end > 100:
            columns.append((last_end, image.width))

        return columns if columns else [(0, image.width)]

    # Если колонки не найдены, пробуем метод разделения пополам
    if vertical_projection[image.width//2] < gap_threshold:
        return [(0, image.width//2 - 20), (image.width//2 + 20, image.width)]

    return [(0, image.width)]

def detect_large_text(image):
    """Обнаружение крупного текста (заголовков)"""
    # Конвертация в grayscale
    img = np.array(image.convert('L'))

    # Бинаризация
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_text_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Фильтруем маленькие области и слишком большие (картинки)
        if w < 50 or h < 30 or w > image.width * 0.8 or h > image.height * 0.8:
            continue

        # Фильтруем по соотношению сторон (исключаем линии)
        aspect_ratio = w / h
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            continue

        # Фильтруем по размеру (крупный текст)
        if h > 40:  # Минимальная высота для крупного текста
            large_text_areas.append((x, y, x+w, y+h))

    return large_text_areas

def extract_large_text(image, areas):
    """Извлечение текста из областей с крупным шрифтом"""
    large_text = ""

    for area in areas:
        x1, y1, x2, y2 = area
        # Увеличиваем область для лучшего распознавания
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(image.width, x2 + 10)
        y2 = min(image.height, y2 + 10)

        text_img = image.crop((x1, y1, x2, y2))

        # Применяем дополнительные фильтры для улучшения качества
        text_img = text_img.filter(ImageFilter.SHARPEN)

        # Распознаем с параметрами для крупного текста
        text = pytesseract.image_to_string(
            np.array(text_img),
            lang='rus+eng',
            config=(
                '--oem 1 '
                '--psm 6 '  # Единый блок текста
                '-c textord_min_xheight=30 '  # Минимальная высота символов
            )
        )
        large_text += text.strip() + "\n"

    return large_text.strip()

def process_page_content(image, columns):
    """Обработка основного контента страницы"""
    content_text = ""

    for i, (start, end) in enumerate(columns):
        col_img = image.crop((start, 0, end, image.height))

        # Предварительная обработка изображения колонки
        col_img = col_img.filter(ImageFilter.SHARPEN)

        # Распознавание
        text = pytesseract.image_to_string(
            np.array(col_img),
            lang='rus+eng',
            config=(
                '--oem 1 '
                '--psm 6 '
                '-c preserve_interword_spaces=1 '
                '-c textord_tabfind_find_tables=0 '
                '-c textord_noise_normratio=0.5 '
                '-c textord_min_linesize=2.0 '
            )
        )
        content_text += text + "\n\n"

    return content_text

def preprocess_image(image):
    """Улучшение качества изображения с акцентом на текст"""
    img = np.array(image)

    # Увеличение контраста с использованием CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Уменьшение шума
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, None, 10, 10, 7, 21
    )

    # Увеличение резкости
    sharpened = cv2.filter2D(denoised, -1, np.array([[-1, -1, -1],
                                                    [-1, 9, -1],
                                                    [-1, -1, -1]]))

    return Image.fromarray(sharpened)

def debug_visualization(image, columns, large_text_areas, page_num):
    """Создание отладочных изображений с визуализацией"""
    debug_img = np.array(image.copy())

    # Рисуем колонки
    for i, (start, end) in enumerate(columns):
        color = (255, 0, 0) if i % 2 == 0 else (0, 255, 0)
        cv2.line(debug_img, (start, 0), (start, debug_img.shape[0]), color, 2)
        cv2.line(debug_img, (end, 0), (end, debug_img.shape[0]), color, 2)
        cv2.putText(debug_img, f"Col {i+1}", (start+10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Рисуем области крупного текста
    for x1, y1, x2, y2 in large_text_areas:
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(debug_img, "Large Text", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Сохраняем
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_path = os.path.join(DEBUG_DIR, f"page_{page_num:03d}_layout.jpg")
    Image.fromarray(debug_img).save(debug_path)
    log(f"Сохранено отладочное изображение: {debug_path}")

def process_page(image, page_num):
    """Обработка одной страницы"""
    # Предобработка изображения
    processed_image = preprocess_image(image)

    # Определение колонок
    columns = detect_columns(processed_image)
    log(f"На странице обнаружено колонок: {len(columns)}")

    # Поиск крупного текста (заголовков)
    large_text_areas = detect_large_text(processed_image)
    log(f"На странице обнаружено областей крупного текста: {len(large_text_areas)}")

    # Визуализация для отладки
    debug_visualization(processed_image, columns, large_text_areas, page_num)

    # Извлечение крупного текста
    large_text = extract_large_text(processed_image, large_text_areas)

    # Обработка основного контента
    content_text = process_page_content(processed_image, columns)

    return large_text + "\n\n" + content_text

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

            # Обработка страницы
            raw_text = process_page(page, i)

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
    # Для macOS
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

    # Дополнительные параметры Tesseract
    os.environ['OMP_THREAD_LIMIT'] = '1'

    process_book()
