import sys
sys.modules['google.cloud.vision'] = None  # Отключаем проблемный импорт

import pdf2image
from layoutparser.ocr import TesseractAgent
from layoutparser.models import Detectron2LayoutModel
import layoutparser as lp
import cv2
import numpy as np

# 1. Конвертация PDF в изображение
images = pdf2image.convert_from_path(
    "Снимок экрана 2025-07-16 в 15.22.pdf",
    dpi=600,
    poppler_path="/usr/local/bin"  # Путь к poppler для Mac
)

# 2. Инициализация OCR агента для русского
ocr_agent = TesseractAgent(
    languages="rus",  # русский язык
    config="--psm 6 --oem 3 -c preserve_interword_spaces=1"  # настройки для лучшего распознавания
)

# 3. Инициализация модели для детекции блоков
model = Detectron2LayoutModel(
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],  # порог уверенности
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}  # карта меток
)

all_text = []  # Для хранения текста всех страниц

for i, pil_image in enumerate(images):
    print(f"Обработка страницы {i+1}...")

    # Конвертация в формат OpenCV
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 4. Детекция текстовых блоков
    layout = model.detect(image)

    # Фильтруем только текстовые блоки
    text_blocks = [b for b in layout if b.type in ["Text", "Title", "List"]]

    # 5. Распознавание текста в каждом блоке
    page_text = []
    for block in text_blocks:
        # Вырезаем блок с отступами
        cropped_image = block.pad(left=15, right=15, top=5, bottom=5).crop_image(image)

        # Распознаем текст
        text = ocr_agent.detect(cropped_image)

        # Сохраняем результат
        block.set(text=text, inplace=True)

        # Форматируем по типу блока
        if block.type == "Title":
            page_text.append(f"\n# {text.strip()}\n")
        elif block.type == "List":
            items = [f"- {item.strip()}" for item in text.split('\n') if item.strip()]
            page_text.append("\n".join(items))
        else:
            page_text.append(text.strip())

    # Добавляем текст страницы в общий результат
    all_text.append("\n\n".join(page_text))
    print(f"Страница {i+1} обработана, блоков: {len(text_blocks)}")

# 6. Сохраняем результат
with open("output1.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print("Распознавание завершено успешно! Результат сохранён в output.txt")
