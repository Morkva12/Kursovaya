import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

Image.MAX_IMAGE_PIXELS = 500_000_000

# Загрузка предобученных моделей YOLOv8
model_text = YOLO("best.pt")
model_segmentation = YOLO('Sbest.pt')
combined_mask_global = None
img_preview_path = None
filepath = None

simple_lama = SimpleLama()

# Переменные для регулировки добавочных пикселей
text_padding = 10
sound_padding = 10

# Переменные для хранения масок
mask_sound = None
mask_text = None

# Переменная для включения экспериментального режима
experimental_mode = None


def apply_masks(image_path, options, text_padding, sound_padding):
    global combined_mask_global, mask_sound, mask_text
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Создание копий масок с padding
    mask_sound_padded = np.zeros((h, w), dtype=np.uint8)
    if mask_sound is not None and "sound" in options:
        mask_sound_padded = cv2.dilate(mask_sound, np.ones((sound_padding, sound_padding), np.uint8), iterations=1)

    mask_text_padded = np.zeros((h, w), dtype=np.uint8)
    if mask_text is not None and "text" in options:
        mask_text_padded = np.copy(mask_text)
        if text_padding > 0:
            kernel = np.ones((text_padding, text_padding), np.uint8)
            mask_text_padded = cv2.dilate(mask_text_padded, kernel, iterations=1)

    # Проверка и удаление перекрывающихся областей
    if "sound" in options and "text" in options:
        intersection = cv2.bitwise_and(mask_sound_padded, mask_text_padded)
        mask_text_padded[intersection == 255] = 0

    # Объединение масок
    combined_mask_global = cv2.bitwise_or(mask_text_padded, mask_sound_padded)

    # Наложение красных масок на изображение
    img_with_masks = img.copy()
    img_with_masks[combined_mask_global == 255] = [0, 0, 255]

    # Сохранение изображения с масками
    cv2.imwrite("image_with_masks.png", img_with_masks)

    return "image_with_masks.png"


def remove_mask():
    global combined_mask_global
    if filepath and combined_mask_global is not None:
        img = Image.open(filepath).convert('RGB')

        # Преобразование изображения PIL в массив numpy
        img_np = np.array(img)

        # Использование глобальной объединённой маски
        combined_mask = combined_mask_global

        # Преобразование массива numpy обратно в изображение PIL
        combined_mask_pil = Image.fromarray(combined_mask)

        # Использование simple_lama для удаления областей, обозначенных маской
        result = simple_lama(img, combined_mask_pil)

        # Сохранение результата и обновление предварительного просмотра
        result.save("inpainted.png")
        update_preview("inpainted.png")


def load_photo():
    global filepath, mask_sound, mask_text
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        print(f"Selected file: {filepath}")
        mask_sound, mask_text = generate_initial_masks(filepath)
        update_preview(filepath)


def generate_initial_masks(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    mask_sound = np.zeros((h, w), dtype=np.uint8)
    mask_text = np.zeros((h, w), dtype=np.uint8)

    results_segmentation = model_segmentation(image_path)
    if results_segmentation[0].masks is not None:
        for segment in results_segmentation[0].masks.xy:
            poly = np.array(segment, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_sound, [poly], 255)

    results_text = model_text(image_path)
    for box in results_text[0].boxes:
        coords = box.xyxy.cpu().numpy().flatten()
        x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(mask_text, (x1, y1), (x2, y2), 255, -1)

    return mask_sound, mask_text


def update_preview(image_path):
    global img_preview_path
    if image_path:
        img = Image.open(image_path)
        img.thumbnail((1000, 1000))
        img_preview = ImageTk.PhotoImage(img)
        preview_label.configure(image=img_preview)
        preview_label.image = img_preview
        img_preview_path = image_path  # Сохранение пути к изображению для последующего сохранения


def save_image():
    if img_preview_path:
        # Открытие диалогового окна для выбора места сохранения файла
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg;*.jpeg"),
                                                            ("All files", "*.*")])
        if save_path:
            # Сохранение изображения по выбранному пути
            Image.open(img_preview_path).save(save_path)
            messagebox.showinfo("Info", f"Image saved to {save_path}")


def on_checkbox_changed():
    if filepath:
        options = []
        if text_var.get():
            options.append("text")
        if sound_var.get():
            options.append("sound")
        updated_image_path = apply_masks(filepath, options, text_padding, sound_padding)
        update_preview(updated_image_path)
        if experimental_mode:
            update_debug_info()


def on_text_slider_changed(val):
    global text_padding
    text_padding = int(val)
    if filepath:
        on_checkbox_changed()
    if experimental_mode:
        text_debug_label.config(text=f"Text Mask Padding: {text_padding}")


def on_sound_slider_changed(val):
    global sound_padding
    sound_padding = int(val)
    if filepath:
        on_checkbox_changed()
    if experimental_mode:
        sound_debug_label.config(text=f"Sound Mask Padding: {sound_padding}")


def update_debug_info():
    text_debug_label.config(text=f"Text Mask Padding: {text_padding}")
    sound_debug_label.config(text=f"Sound Mask Padding: {sound_padding}")


window = tk.Tk()
window.title("YOLOv8 Segmenter")

load_button = tk.Button(window, text="Load", command=load_photo)
load_button.pack()

text_var = tk.BooleanVar()
sound_var = tk.BooleanVar()

checkbox_text = tk.Checkbutton(window, text="Text", variable=text_var, command=on_checkbox_changed)
checkbox_text.pack()

checkbox_sound = tk.Checkbutton(window, text="Sound", variable=sound_var, command=on_checkbox_changed)
checkbox_sound.pack()

remove_button = tk.Button(window, text="Remove", command=remove_mask)
remove_button.pack()

save_button = tk.Button(window, text="Save", command=save_image)
save_button.pack()

preview_label = tk.Label(window)
preview_label.pack()

if experimental_mode:
    text_debug_label = tk.Label(window, text=f"Text Mask Padding: {text_padding}")
    text_debug_label.pack()

    sound_debug_label = tk.Label(window, text=f"Sound Mask Padding: {sound_padding}")
    sound_debug_label.pack()

window.mainloop()
