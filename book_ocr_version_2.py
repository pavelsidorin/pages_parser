import os
import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from datetime import datetime
from PIL import Image
import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
from spellchecker import SpellChecker
import pandas as pd

class BookProcessor:
    """Класс для обработки PDF-книг с улучшенной предобработкой и орфографией."""

    def __init__(self, input_pdf, output_dir="parsed_book", dpi=300, column_margin=50, debug_page=None):
        """Инициализация с параметрами конфигурации."""
        self.input_pdf = input_pdf
        self.output_dir = output_dir
        self.dpi = dpi
        self.column_margin = column_margin
        self.debug_page = debug_page
        self.logger = self._setup_logger()
        self.spell_checker_ru = SpellChecker(language='ru')
        self.spell_checker_en = SpellChecker(language='en')
        self.custom_dict = {'яффе': 'Яффе', 'фенвик': 'Фенвик', 'секс': 'секс', 'женцины': 'женщины', 'сексуалный': 'сексуальный'}

        if not os.path.exists(input_pdf):
            raise FileNotFoundError(f"Входной PDF {input_pdf} не существует")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "debug"), exist_ok=True)

    def _setup_logger(self):
        """Настройка логирования."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        return logging.getLogger()

    def correct_spelling(self, text):
        """Исправление орфографии в тексте."""
        words = text.split()
        corrected_words = []

        for word in words:
            if re.match(r'[а-яА-Я]', word):
                corrected = self.custom_dict.get(word.lower(), self.spell_checker_ru.correction(word) or word)
            else:
                corrected = self.custom_dict.get(word.lower(), self.spell_checker_en.correction(word) or word)
            corrected_words.append(corrected)

        return ' '.join(corrected_words)

    def normalize_text(self, text):
        """Очистка и нормализация текста с исправлением орфографии."""
        text = re.sub(r'(\b[и3]\s*){2,}', ' ', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'-\s*\n', '', text)
        text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = self.correct_spelling(text.strip())
        return text

    def detect_columns(self, image):
        """Обнаружение столбцов с улучшенной обработкой."""
        img = np.array(image.convert('L'))
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Попытка обнаружения вертикальных линий
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=img.shape[0]//3, maxLineGap=20)

        columns = []
        if lines is not None:
            self.logger.info("Обнаружены линии Хафа")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 15:  # Более мягкое условие для вертикальных линий
                    columns.append((x1, x1 + self.column_margin))

        # Если линии не найдены, используем проекцию пикселей
        if not columns:
            self.logger.info("Линии не найдены, используется проекция пикселей")
            vertical_projection = np.sum(binary, axis=0)
            in_column, start_idx = False, 0
            for idx, value in enumerate(vertical_projection):
                if value > 0 and not in_column:
                    in_column = True
                    start_idx = idx
                elif value == 0 and in_column and idx - start_idx > self.column_margin:
                    in_column = False
                    columns.append((start_idx, idx))

        # Если столбцы не найдены, предполагаем одну колонку
        if not columns:
            self.logger.info("Столбцы не найдены, используется одна колонка")
            width = image.width
            columns = [(0, width)]

        # Обеспечиваем минимум две колонки, если ожидается двухколоночный макет
        if len(columns) < 2:
            width = image.width
            columns = [(0, width//2 - self.column_margin//2), (width//2 + self.column_margin//2, width)]

        self.logger.info(f"Обнаружено столбцов: {len(columns)}")
        return columns

    def preprocess_image(self, image):
        """Улучшение качества изображения для OCR."""
        scale_factor = 300 / self.dpi
        img = np.array(image)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
        img[mask > 0] = [255, 255, 255]

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        kernel = np.ones((3, 3), np.uint8)
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return Image.fromarray(denoised)

    def process_column(self, image, start, end):
        """Извлечение текста из одной колонки с учетом таблиц."""
        column_img = image.crop((max(0, start-10), 0, min(image.width, end+10), image.height))
        data = pytesseract.image_to_data(
            column_img,
            lang='rus+eng',
            config='--oem 1 --psm 6 -c preserve_interword_spaces=1 -c textord_tabfind_find_tables=1',
            output_type=pytesseract.Output.DATAFRAME
        )

        table_text = []
        for _, row in data.iterrows():
            if row['conf'] > 50:
                table_text.append(row['text'])

        return "\n".join([t for t in table_text if t and isinstance(t, str)])

    def save_debug_image(self, page_num, image, columns):
        """Сохранение отладочного изображения с границами столбцов."""
        debug_img = np.array(image.copy())
        for idx, (start, end) in enumerate(columns):
            cv2.line(debug_img, (start, 0), (start, debug_img.shape[0]), (255,0,0), 2)
            cv2.line(debug_img, (end, 0), (end, debug_img.shape[0]), (0,255,0), 2)
            cv2.putText(debug_img, f"Col {idx+1}", (start+10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        Image.fromarray(debug_img).save(
            os.path.join(self.output_dir, "debug", f"page_{page_num:03d}_columns.jpg")
        )

    def process_page(self, page, page_num, total_pages):
        """Обработка одной страницы."""
        try:
            self.logger.info(f"Обработка страницы {page_num}/{total_pages}")
            processed_page = self.preprocess_image(page)
            columns = self.detect_columns(processed_page)

            if self.debug_page is None or page_num == self.debug_page:
                self.save_debug_image(page_num, processed_page, columns)

            column_texts = [self.process_column(processed_page, start, end) for start, end in columns]
            raw_text = "\n\n".join(column_texts)
            clean_text = self.normalize_text(raw_text)

            with open(os.path.join(self.output_dir, f"page_{page_num:03d}.txt"), 'w', encoding='utf-8') as f:
                f.write(clean_text)

            return f"--- СТРАНИЦА {page_num} ---\n{clean_text}\n\n"

        except Exception as e:
            self.logger.error(f"Ошибка на странице {page_num}: {str(e)}")
            return ""

    def process(self, max_workers=4):
        """Обработка всего PDF-документа."""
        self.logger.info(f"Начало обработки: {self.input_pdf}")

        try:
            pages = convert_from_path(self.input_pdf, dpi=self.dpi)
        except Exception as e:
            self.logger.error(f"Ошибка конвертации: {str(e)}")
            return

        full_text = ""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_page, page, i+1, len(pages))
                for i, page in enumerate(pages)
            ]
            for future in futures:
                full_text += future.result()

        with open(os.path.join(self.output_dir, "full_book.txt"), 'w', encoding='utf-8') as f:
            f.write(full_text)

        self.logger.info(f"Обработка завершена! Результаты в: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Обработка PDF-книги и извлечение текста.")
    parser.add_argument("input_pdf", help="Путь к входному PDF-файлу")
    parser.add_argument("--output-dir", default="parsed_book", help="Папка для результатов")
    parser.add_argument("--dpi", type=int, default=300, help="DPI для конвертации PDF")
    parser.add_argument("--column-margin", type=int, default=50, help="Минимальный зазор между колонками")
    parser.add_argument("--debug-page", type=int, help="Номер страницы для отладочного изображения")
    args = parser.parse_args()

    processor = BookProcessor(
        args.input_pdf,
        args.output_dir,
        args.dpi,
        args.column_margin,
        args.debug_page
    )
    processor.process()

if __name__ == "__main__":
    main()
