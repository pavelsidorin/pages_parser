import os
import re
import numpy as np
from pdf2image import convert_from_path
from datetime import datetime
import pytesseract
import nltk
from torch.utils.data import Dataset
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import argparse

OUTPUT_DIR = "parsed_book"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def normalize_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(\S)\n(\w)', r'\1 \2', text)
    text = re.sub(r'\n{2,}', '\n', text)
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
        source = item['original']  # Ошибочный текст
        target = item['normalized']  # Нормализованный текст
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
    if torch.backends.mps.is_available():
        model = model.to("mps")
        log("Модель перенесена на MPS (GPU)")
    else:
        log("Используется CPU")
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
    log(f"Модель сохранена в {output_dir}")

def init_normalization_model(model_path="./fine_tuned_rut5"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        log(f"Загрузка модели: {model_path}")
        return model, tokenizer
    except Exception as e:
        log(f"Не получилось загрузить модель: {e}")
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

        input_text = f"нормализуй: {cleaned_sentence}"
        try:
            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True
            ).input_ids
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
            log(f"Ошибка обработки '{cleaned_sentence}': {e}")
            normalized_sentences.append(cleaned_sentence)

    return '\n'.join(normalized_sentences)

def process_pdf(pdf_path, model, tokenizer):
    pages = convert_from_path(pdf_path, dpi=600, poppler_path=r'/usr/local/bin')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("Начало обработки: pdf")
    for i, page in enumerate(pages):
        log(f"Обработка страницы {i + 1}/{len(pages)}")
        text = pytesseract.image_to_string(page, lang='rus+eng')
        text = normalize_text(text)
        text = smart_text_normalization(text, model, tokenizer)
        with open(os.path.join(OUTPUT_DIR, f"page_{i + 1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

def main():
    nltk.download('punkt', quiet=True)
    parser = argparse.ArgumentParser(description="PDF text extraction and normalization")
    parser.add_argument("--study", action="store_true", help="Train the normalization model")
    args = parser.parse_args()

    dataset_path = "paraphrase_dataset.json"
    model_path = "./fine_tuned_rut5"

    if args.study:
        train_normalization_model(
            model_name="ai-forever/ruT5-base",
            data_path=dataset_path,
            output_dir=model_path,
            num_epochs=5
        )

    model, tokenizer = init_normalization_model(model_path)
    process_pdf("Менталист.pdf", model, tokenizer)

if __name__ == "__main__":
    main()
