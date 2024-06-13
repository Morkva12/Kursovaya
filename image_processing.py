# image_processing.py
"""
Эта программа выполняет обработку изображений с использованием предобученных моделей YOLOv8 и Simple LaMa.
Основные функции программы включают:
1. Генерация начальных масок для текста и звука на изображении.
2. Применение масок с возможностью расширения области маски.
3. Удаление областей масок с изображений с использованием метода inpainting Simple LaMa.
"""

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

# Загрузка предобученных моделей YOLOv8 для текста и сегментации
model_text = YOLO("best.pt") # Модель для детекции текста
model_segmentation = YOLO('Sbest.pt') # Модель для детекции звуков
simple_lama = SimpleLama()

def apply_masks(image_path, options, text_padding, sound_padding, mask_sound, mask_text):
    """
    Применяет маски текста и звука к изображению с учетом заданных параметров расширения масок.

    :param image_path: путь к изображению
    :param options: опции, определяющие, какие маски применять (sound, text)
    :param text_padding: количество пикселей для расширения области маски текста
    :param sound_padding: количество пикселей для расширения области маски звука
    :param mask_sound: маска звука
    :param mask_text: маска текста
    :return: путь к изображению с наложенными масками, комбинированная маска
    """
    # Чтение изображения
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    print("Начало процесса применения масок")
    print(f"Размеры изображения: {h}x{w}")
    print(f"Опции: {options}")
    print(f"Увеличение области маски текста: {text_padding}, Увеличение области маски звука: {sound_padding}")

    # Создание копий масок с увеличенной областью для звука
    mask_sound_padded = np.zeros((h, w), dtype=np.uint8)
    if mask_sound is not None and "sound" in options:
        mask_sound_padded = cv2.dilate(mask_sound, np.ones((sound_padding, sound_padding), np.uint8), iterations=1)
        print("Применено увеличение области маски звука")

    # Создание копий масок с увеличенной областью для текста
    mask_text_padded = np.zeros((h, w), dtype=np.uint8)
    if mask_text is not None and "text" in options:
        mask_text_padded = np.copy(mask_text)
        if text_padding > 0:
            kernel = np.ones((text_padding, text_padding), np.uint8)
            mask_text_padded = cv2.dilate(mask_text_padded, kernel, iterations=1)
        print("Применено увеличение области маски текста")

    # Объединение масок текста и звука в одну комбинированную маску
    combined_mask_global = cv2.bitwise_or(mask_text_padded, mask_sound_padded)
    print(f"Создана комбинированная маска. Комбинированная маска None: {combined_mask_global is None}")

    # Наложение красных масок на исходное изображение
    img_with_masks = img.copy()
    img_with_masks[combined_mask_global == 255] = [0, 0, 255]
    print("Красные маски наложены на изображение")

    # Сохранение изображения с наложенными масками
    image_with_masks_path = "image_with_masks.png"
    cv2.imwrite(image_with_masks_path, img_with_masks)
    print(f"Изображение с масками сохранено в {image_with_masks_path}")

    return image_with_masks_path, combined_mask_global

def remove_mask_with_lama(filepath, combined_mask):
    """
    Удаляет области масок с изображения с использованием метода inpainting от Simple LaMa.

    :param filepath: путь к изображению с наложенными масками
    :param combined_mask: комбинированная маска текста и звука
    :return: путь к изображению после инпейнтинга
    """
    print("Начало удаления масок с помощью LaMa")
    if filepath and combined_mask is not None:
        # Открытие изображения и преобразование маски в формат PIL
        img = Image.open(filepath).convert('RGB')
        img_np = np.array(img)
        combined_mask_pil = Image.fromarray(combined_mask)

        # Применение Simple LaMa для удаления масок
        result = simple_lama(img, combined_mask_pil)
        result_path = "inpainted.png"
        result.save(result_path)
        print(f"Результат инпейнтинга сохранен в {result_path}")
        return result_path
    else:
        print("Не найдено действительного файла или комбинированной маски для удаления")

def generate_initial_masks(image_path):
    """
    Генерирует начальные маски для текста и звука на изображении.

    :param image_path: путь к изображению
    :return: маска звука, маска текста
    """
    # Чтение изображения
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    print(f"Генерация начальных масок для изображения {image_path}")

    # Инициализация пустых масок для звука и текста
    mask_sound = np.zeros((h, w), dtype=np.uint8)
    mask_text = np.zeros((h, w), dtype=np.uint8)

    # Получение результатов сегментации для звука
    results_segmentation = model_segmentation(image_path)
    if results_segmentation[0].masks is not None:
        for segment in results_segmentation[0].masks.xy:
            poly = np.array(segment, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_sound, [poly], 255)
        print("Сгенерирована маска звука")

    # Получение результатов для текста
    results_text = model_text(image_path)
    for box in results_text[0].boxes:
        coords = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(mask_text, (x1, y1), (x2, y2), 255, -1)
    print("Сгенерирована маска текста")

    return mask_sound, mask_text
