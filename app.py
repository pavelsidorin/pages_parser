import os
import re
import numpy as np
from pdf2image import convert_from_path
from datetime import datetime
import pytesseract
from spellchecker import SpellChecker
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from PIL import Image, ImageEnhance
from pymorphy3 import MorphAnalyzer
import nltk
from torch.utils.data import Dataset
import json
import argparse
from sentence_transformers import SentenceTransformer, util

OUTPUT_DIR = "parsed_book"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def normalize_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\n\1\2', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1-\2\n', text)
    text = re.sub(r'(\S)\n(\w)', r'\1 \n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def remove_mixed_words(text):
    tokens = re.findall(r'\b\S+\b|[^\w\s]', text)
    cleaned_tokens = []
    for token in tokens:
        if not re.match(r'^\w', token):
            cleaned_tokens.append(token)
            continue
        if re.search(r'[^а-яА-Яa-zA-Z0-9\'-]', token):
            continue
        has_russian = re.search(r'[а-яА-Я]', token)
        has_english = re.search(r'[a-zA-Z]', token)
        if has_russian and has_english:
            continue
        cleaned_tokens.append(token)
    return ' '.join(cleaned_tokens)

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?:;()"«»\n-]', '', text)
    text = re.sub(r'(\w)\1{3,}', r'\1', text)
    text = remove_mixed_words(text)
    return text.strip()

def correct_spelling(text):
    morph = MorphAnalyzer()
    valid_words = set()
    tokens = re.findall(r'\b[\w]+\b', text)
    for token in tokens:
        parsed = morph.parse(token)[0]
        if parsed.score > 0.5:
            valid_words.add(token.lower())
    lines = nltk.sent_tokenize(text, language='russian')
    corrected_lines = []
    for line in lines:
        words = re.split(r'(\b[\w]+\b)', line)
        corrected_words = []
        for word in words:
            if re.fullmatch(r'\w+', word):
                if word.lower() not in valid_words:
                    parsed = morph.parse(word)[0]
                    corrected = parsed.normal_form
                    if word.istitle():
                        corrected = corrected.title()
                    elif word.isupper():
                        corrected = corrected.upper()
                    elif word[0].isupper():
                        corrected = corrected.capitalize()
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        corrected_lines.append(''.join(corrected_words))
    return '\n'.join(corrected_lines)

def init_paraphrase_model(model_path="./fine_tuned_rut5"):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        log(f"Загрузка модели: {model_path}")
        return model, tokenizer
    except Exception as e:
        log(f"Не получилось загрузить модель: {e}")
        raise

def init_sentence_bert():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def smart_sentence_correction(text, model, tokenizer, sentence_bert, max_length=512):
    sentences = nltk.sent_tokenize(text, language='russian')
    corrected_sentences = []
    similarity_threshold = 0.7  # Порог косинусного сходства

    for sentence in sentences:
        if len(sentence.strip()) < 5 or re.match(r'^\s*[.,!?]+\s*$', sentence):
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
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Проверяем семантическую близость
            if len(corrected.split()) >= 5 and re.search(r'[а-яА-Яa-zA-Z]', corrected):
                embedding1 = sentence_bert.encode(sentence, convert_to_tensor=True)
                embedding2 = sentence_bert.encode(corrected, convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
                if cosine_score < similarity_threshold:
                    corrected_sentences.append(sentence)  # Возвращаем оригинал, если сходство низкое
                else:
                    corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sentence)  # Возвращаем оригинал, если перефразирование неудачное
        except Exception as e:
            log(f"Ошибка обработки '{cleaned_sentence}': {e}")
            corrected_sentences.append(sentence)

    return '\n'.join(corrected_sentences)

class ParaphraseDataset(Dataset):
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
        target = item['paraphrase']
        source_encoding = self.tokenizer(
            source,
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

def train_paraphrase_model(model_name, data_path, output_dir, num_epochs=3):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    if torch.backends.mps.is_available():
        model = model.to("mps")
        log("Модель перенесена на MPS (GPU)")
    else:
        log("Используется CPU")
    dataset = ParaphraseDataset(tokenizer, data_path)
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

def remove_unrecoverable_words(text):
    morph = MorphAnalyzer()
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        if re.match(r'\w+', token):
            parsed = morph.parse(token)[0]
            if parsed.score < 0.05:
                continue
        cleaned_tokens.append(token)
    return ' '.join(cleaned_tokens)

def process_pdf(pdf_path, model, tokenizer, sentence_bert):
    pages = convert_from_path(pdf_path, dpi=600, poppler_path=r'/usr/local/bin')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("Начало обработки: pdf")
    for i, page in enumerate(pages):
        log(f"Обработка страницы {i + 1}/{len(pages)}")
        text = pytesseract.image_to_string(page, lang='rus+eng')
        text = normalize_text(text)
        # text = correct_spelling(text)
        # text = smart_sentence_correction(text, model, tokenizer, sentence_bert)
        with open(os.path.join(OUTPUT_DIR, f"page_{i + 1}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

def main():
    parser = argparse.ArgumentParser(description="PDF text extraction and processing")
    parser.add_argument("--study", action="store_true", help="Train the paraphrase model")
    args = parser.parse_args()

    nltk.download('punkt', quiet=True)
    dataset_path = "paraphrase_dataset.json"
    model_path = "./fine_tuned_rut5"

    if args.study:
        train_paraphrase_model(
            model_name="cointegrated/rut5-base-paraphraser",
            data_path=dataset_path,
            output_dir=model_path,
            num_epochs=3
        )

    paraphrase_model, paraphrase_tokenizer = init_paraphrase_model(model_path)
    sentence_bert = init_sentence_bert()
    process_pdf("Снимок экрана 2025-07-16 в 15.22.pdf", paraphrase_model, paraphrase_tokenizer, sentence_bert)

if __name__ == "__main__":
    main()
