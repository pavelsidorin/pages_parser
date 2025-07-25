import sys
# Убираем блокировку Google Cloud Vision
if 'google.cloud.vision' in sys.modules:
    del sys.modules['google.cloud.vision']

import PIL.Image
if not hasattr(PIL.Image, 'LINEAR'):
    PIL.Image.LINEAR = PIL.Image.BILINEAR
import os
import re
import json
import argparse
import subprocess
import numpy as np
from datetime import datetime
import cv2
import nltk
import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import pdf2image
from layoutparser.ocr import TesseractAgent
from layoutparser.models import Detectron2LayoutModel
import layoutparser as lp
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError, RetryError

OUTPUT_DIR = "parsed_book"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def normalize_text(text):
    text = re.sub(r'(\w+)-\n(\w+)', r'\n\1\2', text)
    text = re.sub(r'(\w+)-\n\n(\w+)', r'\n\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?:;()"«»\n-]', '', text)
    text = re.sub(r'\b[a-zA-Z]+\d+\b|\b\d+[a-zA-Z]+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[a-zA-Z]{2,}\b(?![а-яА-Я])', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class NormalizationDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['original']
        target = item['normalized']
        source_encoding = self.tokenizer(
            f"нормализуй: {source}",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def train_normalization_model(model_name, data_path, output_dir, num_epochs=5):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    log(f"Модель перенесена на {device.upper()}")

    dataset = NormalizationDataset(tokenizer, data_path)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"Модель нормализации сохранена в {output_dir}")

def init_normalization_model(model_path="./fine_tuned_rut5"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        log(f"Загрузка модели нормализации: {model_path}")
        return model, tokenizer
    except Exception as e:
        log(f"Не получилось загрузить модель нормализации: {e}")
        raise

def smart_text_normalization(text, model, tokenizer, max_length=512):
    sentences = nltk.sent_tokenize(text, language='russian')
    normalized_sentences = []

    for sentence in sentences:
        if len(sentence.strip()) < 5 or not re.search(r'[а-яА-Я]', sentence):
            continue

        cleaned_sentence = clean_text(sentence)
        if not cleaned_sentence.strip():
            continue

        input_text = cleaned_sentence
        try:
            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True
            ).input_ids

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            input_ids = input_ids.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            normalized = tokenizer.decode(outputs[0], skip_special_tokens=True)
            normalized_sentences.append(normalized)
        except Exception as e:
            log(f"Ошибка обработки предложения: {e}")
            normalized_sentences.append(cleaned_sentence)

    return '\n'.join(normalized_sentences)

def init_layout_parser():
    log("Инициализация LayoutParser...")

    try:
        poppler_path = subprocess.check_output("which pdfinfo", shell=True).decode().strip()
        poppler_path = poppler_path.rsplit('/', 1)[0]
        log(f"✅ Найден poppler в: {poppler_path}")
    except Exception as e:
        poppler_path = None
        log(f"⚠️ Не удалось найти poppler: {e}. Используем системный PATH")

    ocr_agent = TesseractAgent(
        languages="rus",
        # config="--psm 6 --oem 3 -c preserve_interword_spaces=1"
    )

    try:
        model = Detectron2LayoutModel(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        log("✅ Модель детекции блоков успешно загружена")
    except Exception as e:
        log(f"⚠️ Ошибка загрузки модели детекции блоков: {e}")
        model = None

    return ocr_agent, model, poppler_path

def sort_with_cloud_vision(image, layout_blocks):
    """Использует Google Cloud Vision для определения порядка чтения блоков"""
    try:
        # Инициализируем клиент Cloud Vision
        client = vision.ImageAnnotatorClient()

        # Конвертируем изображение в формат, понятный Cloud Vision
        _, encoded_image = cv2.imencode('.png', image)
        content = encoded_image.tobytes()
        vision_image = vision.Image(content=content)

        # Получаем аннотации документа
        response = client.document_text_detection(image=vision_image)
        document = response.full_text_annotation

        # Сопоставляем блоки LayoutParser с блоками Cloud Vision
        sorted_blocks = []
        visited_blocks = set()

        # Проходим по всем страницам, блокам и параграфам в порядке чтения
        for page in document.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    # Получаем ограничивающий прямоугольник параграфа
                    vertices = [(vertex.x, vertex.y) for vertex in paragraph.bounding_box.vertices]

                    # Ищем соответствующий блок в LayoutParser
                    best_match = None
                    best_iou = 0.0

                    for layout_block in layout_blocks:
                        if id(layout_block) in visited_blocks:
                            continue

                        # Сравниваем области пересечения
                        iou = calculate_iou(vertices, layout_block.block)
                        if iou > best_iou and iou > 0.3:  # Порог перекрытия 30%
                            best_iou = iou
                            best_match = layout_block

                    if best_match:
                        sorted_blocks.append(best_match)
                        visited_blocks.add(id(best_match))

        # Добавляем блоки, которые не были сопоставлены
        for block in layout_blocks:
            if id(block) not in visited_blocks:
                sorted_blocks.append(block)

        log(f"✅ Cloud Vision отсортировал {len(sorted_blocks)} блоков")
        return sorted_blocks

    except (GoogleAPICallError, RetryError) as e:
        log(f"⚠️ Ошибка Cloud Vision API: {e}")
    except Exception as e:
        log(f"⚠️ Неожиданная ошибка в Cloud Vision: {e}")

    # В случае ошибки возвращаем исходный порядок
    return layout_blocks

def calculate_iou(vertices, block):
    """Вычисляет степень перекрытия между полигоном Cloud Vision и блоком LayoutParser"""
    try:
        # Создаем контур для полигона Cloud Vision
        poly_contour = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))

        # Создаем контур для прямоугольника LayoutParser
        block_rect = np.array([
            [block.x_1, block.y_1],
            [block.x_2, block.y_1],
            [block.x_2, block.y_2],
            [block.x_1, block.y_2]
        ], dtype=np.int32).reshape((-1, 1, 2))

        # Вычисляем площадь пересечения
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Размер не важен, нужна только форма
        poly_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly_contour], 1)

        block_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(block_mask, [block_rect], 1)

        intersection = np.logical_and(poly_mask, block_mask).sum()
        union = np.logical_or(poly_mask, block_mask).sum()

        return intersection / union if union > 0 else 0.0

    except Exception:
        return 0.0

def sort_layout_items(layout, image_width, image=None):
    """Сортировка блоков с приоритетом для Cloud Vision"""
    # Пытаемся использовать Cloud Vision, если доступно изображение
    if image is not None:
        try:
            return sort_with_cloud_vision(image, layout)
        except Exception as e:
            log(f"⚠️ Ошибка сортировки Cloud Vision: {e}")

    # Резервный метод: геометрическая сортировка
    try:
        # Рассчитываем среднюю высоту блоков
        heights = [block.block.height for block in layout]
        avg_height = sum(heights) / len(heights) if heights else 50
        tolerance = avg_height * 0.4

        # Группируем по строкам
        rows = {}
        for block in layout:
            y_center = block.block.y_1 + block.block.height / 2
            row_key = round(y_center / tolerance) * tolerance

            if row_key not in rows:
                rows[row_key] = []
            rows[row_key].append(block)

        # Сортируем строки
        sorted_rows = []
        for y in sorted(rows.keys()):
            # Сортируем блоки в строке
            row_blocks = sorted(rows[y], key=lambda b: b.block.x_1)
            sorted_rows.append(row_blocks)

        # Собираем финальный порядок
        sorted_blocks = [block for row in sorted_rows for block in row]
        log(f"✅ Геометрическая сортировка: {len(sorted_blocks)} блоков по {len(sorted_rows)} строкам")
        return sorted_blocks

    except Exception as e:
        log(f"⚠️ Ошибка геометрической сортировки: {e}")
        return sorted(layout, key=lambda b: (b.block.y_1, b.block.x_1))

def process_pdf_with_layout(pdf_path, model, tokenizer, ocr_agent, layout_model, poppler_path,
                           use_basic_norm=True, use_smart_norm=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=400,
            poppler_path=poppler_path
        )
        log(f"Успешно конвертировано страниц: {len(images)}")
    except Exception as e:
        log(f"Ошибка конвертации PDF: {e}")
        return

    all_text = []

    for i, pil_image in enumerate(images):
        log(f"Обработка страницы {i+1}/{len(images)} с LayoutParser...")

        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]

        # Детекция блоков
        if layout_model:
            try:
                layout = layout_model.detect(image)
                text_blocks = [b for b in layout if b.type in ["Text", "Title", "List"]]
                log(f"  Найдено блоков: {len(text_blocks)}")
            except Exception as e:
                log(f"⚠️ Ошибка детекции блоков: {e}. Используем полностраничное распознавание")
                text_blocks = [lp.TextBlock(
                    block=lp.Rectangle(0, 0, width, height),
                    type="Text"
                )]
        else:
            text_blocks = [lp.TextBlock(
                block=lp.Rectangle(0, 0, width, height),
                type="Text"
            )]
            log("  Используем полностраничное распознавание")

        # Сортировка блоков с использованием Cloud Vision
        sorted_blocks = sort_layout_items(text_blocks, width, image=image)

        page_text = []
        for j, block in enumerate(sorted_blocks):
            try:
                # Вырезаем и распознаем блок
                cropped_image = block.pad(left=15, right=15, top=10, bottom=10).crop_image(image)
                text = ocr_agent.detect(cropped_image)

                if use_basic_norm:
                    text = normalize_text(text)

                if use_smart_norm:
                    text = normalize_text(text)
                    normalized_text = smart_text_normalization(text, model, tokenizer)
                else:
                    normalized_text = text

                # Форматируем результат
                if block.type == "Title":
                    page_text.append(f"\n# {normalized_text.strip()}\n")
                elif block.type == "List":
                    items = [f"- {item.strip()}" for item in normalized_text.split('\n') if item.strip()]
                    page_text.append("\n".join(items))
                else:
                    page_text.append(normalized_text.strip())

                log(f"  Блок {j+1}/{len(sorted_blocks)} распознан")
            except Exception as e:
                log(f"⚠️ Ошибка обработки блока {j+1}: {e}")
                page_text.append("")

        # Собираем текст страницы
        page_content = "\n\n".join(page_text)
        all_text.append(page_content)

        # Сохраняем страницу
        with open(os.path.join(OUTPUT_DIR, f"page_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(page_content)

    # Сохраняем полный текст
    with open(os.path.join(OUTPUT_DIR, "full_book.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    log("Обработка PDF завершена успешно!")

def main():
    nltk.download('punkt', quiet=True)
    parser = argparse.ArgumentParser(description="PDF text extraction and normalization")
    parser.add_argument("--study", action="store_true", help="Train the normalization model")
    parser.add_argument("--with_smart_normalization", action="store_true",
                        help="Apply smart text normalization with AI model")
    parser.add_argument("--disable_basic_normalization", action="store_true",
                        help="Disable basic text normalization (hyphen removal, etc.)")
    parser.add_argument("--disable_cloud_vision", action="store_true",
                        help="Disable Cloud Vision for block sorting")

    args = parser.parse_args()

    dataset_path = "paraphrase_dataset.json"
    model_path = "./fine_tuned_rut5"
    pdf_path = "Снимок экрана 2025-07-24 в 04.25.pdf"

    # Обучение модели нормализации
    if args.study:
        train_normalization_model(
            model_name="ai-forever/ruT5-base",
            data_path=dataset_path,
            output_dir=model_path,
            num_epochs=3
        )

    # Загрузка модели нормализации
    model, tokenizer = init_normalization_model(model_path)

    # Инициализация LayoutParser
    ocr_agent, layout_model, poppler_path = init_layout_parser()

    # Определение параметров нормализации
    use_basic_norm = not args.disable_basic_normalization
    use_smart_norm = args.with_smart_normalization

    log(f"Параметры обработки: Базовая нормализация={'ВКЛ' if use_basic_norm else 'ВЫКЛ'}, "
        f"Умная нормализация={'ВКЛ' if use_smart_norm else 'ВЫКЛ'}, "
        f"Cloud Vision={'ВЫКЛ' if args.disable_cloud_vision else 'ВКЛ'}")

    # Обработка PDF
    process_pdf_with_layout(
        pdf_path=pdf_path,
        model=model,
        tokenizer=tokenizer,
        ocr_agent=ocr_agent,
        layout_model=layout_model,
        poppler_path=poppler_path,
        use_basic_norm=use_basic_norm,
        use_smart_norm=use_smart_norm
    )

if __name__ == "__main__":
    main()
