import sys
sys.modules['google.cloud.vision'] = None

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

def sort_layout_items(layout, image_width):
    """Умная сортировка блоков с использованием алгоритмов LayoutParser"""
    try:
        # Преобразуем список в объект Layout
        if not isinstance(layout, lp.Layout):
            layout = lp.Layout(layout)

        # Используем встроенную сортировку LayoutParser
        sorted_layout = layout.sort_in_read_order(
            threshold=image_width * 0.1,  # Порог для определения колонок
            key=lambda block: (round(block.block.x_1, -2), round(block.block.y_1, -2))
        )
        return sorted_layout
    except Exception as e:
        # Резервный алгоритм для сложных случаев
        log(f"⚠️ Ошибка встроенной сортировки: {e}. Используем резервный метод")
        return sorted(layout, key=lambda b: (round(b.block.x_1, -2), round(b.block.y_1, -2)))

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

                # Улучшенная сортировка блоков
                text_blocks = sort_layout_items(text_blocks, width)

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

        page_text = []
        for j, block in enumerate(text_blocks):
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

                log(f"  Блок {j+1}/{len(text_blocks)} распознан")
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
    # Новые флаги для управления нормализацией
    parser.add_argument("--with_smart_normalization", action="store_true",
                        help="Apply smart text normalization with AI model")
    parser.add_argument("--disable_basic_normalization", action="store_true",
                        help="Disable basic text normalization (hyphen removal, etc.)")

    args = parser.parse_args()

    dataset_path = "paraphrase_dataset.json"
    model_path = "./fine_tuned_rut5"
    pdf_path = "Снимок экрана 2025-07-16 в 15.22.pdf"

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
        f"Умная нормализация={'ВКЛ' if use_smart_norm else 'ВЫКЛ'}")

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
