"""
Эта программа представляет собой графический интерфейс для обработки изображений с использованием предобученных моделей YOLOv8 и Simple LaMa.
Функции программы включают:
1. Загрузка изображения и генерация начальных масок для текста и звука.
2. Применение масок к изображению с возможностью расширения областей масок.
3. Удаление областей масок с изображений с использованием метода inpainting от Simple LaMa.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
import threading
from image_processing import apply_masks, generate_initial_masks, remove_mask_with_lama
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

Image.MAX_IMAGE_PIXELS = 500_000_000

# Переменные для регулировки добавочных пикселей
text_padding = 10
sound_padding = 10

# Переменные для хранения масок
mask_sound = None
mask_text = None
combined_mask_global = None

# Переменные для хранения путей и состояния
filepath = None
img_preview_path = None
original_img = None
is_processing = False

current_theme = "sandstone"

def switch_theme():
    """
    Переключает тему интерфейса между светлой и темной.
    """
    global current_theme, window
    if current_theme == "sandstone":
        current_theme = "superhero"
        theme_button.config(text="☀️")
    else:
        current_theme = "sandstone"
        theme_button.config(text="🌙")
    window.style.theme_use(current_theme)

# Функции для блокировки и разблокировки виджетов
def lock_widgets():
    """
    Блокирует виджеты интерфейса во время выполнения задачи, чтобы предотвратить
    взаимодействие пользователя с элементами управления.
    """
    global is_processing
    is_processing = True
    load_button.config(state=DISABLED)
    save_button.config(state=DISABLED)
    remove_button.config(state=DISABLED)
    checkbox_text.config(state=DISABLED)
    checkbox_sound.config(state=DISABLED)
    print("Виджеты заблокированы")

def unlock_widgets():
    """
    Разблокирует виджеты интерфейса после завершения задачи, чтобы пользователь мог снова
    взаимодействовать с элементами управления.
    """
    global is_processing
    is_processing = False
    load_button.config(state=NORMAL)
    save_button.config(state=NORMAL if img_preview_path else DISABLED)
    remove_button.config(state=NORMAL if img_preview_path else DISABLED)
    checkbox_text.config(state=NORMAL)
    checkbox_sound.config(state=NORMAL)
    print("Виджеты разблокированы")

def show_loading_indicator():
    """
    Отображает индикатор загрузки в центре окна для индикации процесса обработки.
    """
    progressbar.place(relx=0.5, rely=0.5, anchor=CENTER)
    progressbar.start()

def hide_loading_indicator():
    """
    Скрывает индикатор загрузки после завершения процесса обработки.
    """
    progressbar.stop()
    progressbar.place_forget()

# Функции для обработки изображений
def load_photo():
    """
    Загружает изображение, выбранное пользователем, и начинает процесс генерации масок.
    """
    global filepath, mask_sound, mask_text, original_img
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if filepath:
        print(f"Выбранный файл: {filepath}")
        lock_widgets()
        show_loading_indicator()
        original_img = Image.open(filepath)
        thread = threading.Thread(target=generate_masks_and_update_preview, args=(filepath,))
        thread.start()

def generate_masks_and_update_preview(image_path):
    """
    Генерирует начальные маски для изображения и обновляет предпросмотр.
    """
    global mask_sound, mask_text
    show_processing_image()
    mask_sound, mask_text = generate_initial_masks(image_path)
    options = get_selected_options()
    if options:
        apply_masks_and_update_preview(image_path, options)
    else:
        update_preview(image_path)
        unlock_widgets()
        hide_loading_indicator()

    update_remove_button_state()  # Обновление состояния кнопки Remove после генерации масок

def update_remove_button_state():
    """
    Обновляет состояние кнопки 'Remove' в зависимости от состояния чекбоксов и наличия масок.
    """
    if any([text_var.get(), sound_var.get()]) and (mask_sound is not None or mask_text is not None):
        remove_button.config(state=NORMAL)
    else:
        remove_button.config(state=DISABLED)

def get_selected_options():
    """
    Возвращает список выбранных пользователем опций (текст и/или звук).
    """
    options = []
    if text_var.get():
        options.append("text")
    if sound_var.get():
        options.append("sound")
    return options

def show_processing_image():
    """
    Отображает изображение с эффектом размытия для индикации процесса обработки.
    """
    if original_img:
        img_blurred = original_img.filter(ImageFilter.GaussianBlur(15))
        update_canvas_image(img_blurred, is_blurred=True)


def update_canvas_image(img, is_blurred=False):
    """
    Обновляет изображение на холсте предпросмотра.
    """
    canvas_width = preview_canvas.winfo_width()
    canvas_height = preview_canvas.winfo_height()

    # Сохраняем пропорции изображения
    img_aspect_ratio = img.width / img.height
    canvas_aspect_ratio = canvas_width / canvas_height

    if img_aspect_ratio > canvas_aspect_ratio:
        # Ограничиваем по ширине
        new_width = canvas_width
        new_height = int(new_width / img_aspect_ratio)
    else:
        # Ограничиваем по высоте
        new_height = canvas_height
        new_width = int(new_height * img_aspect_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_preview = ImageTk.PhotoImage(img)

    x = (canvas_width - new_width) // 2
    y = (canvas_height - new_height) // 2

    preview_canvas.create_image(x, y, image=img_preview, anchor="nw")
    preview_canvas.image = img_preview
    preview_canvas.is_blurred = is_blurred
    print("Изображение на холсте обновлено")

def update_preview(image_path):
    """
    Обновляет предпросмотр изображения на основе предоставленного пути к изображению.
    """
    global img_preview_path
    if image_path:
        img_preview_path = image_path
        img = Image.open(image_path)
        update_canvas_image(img)
        print(f"Предпросмотр обновлен с изображением: {image_path}")
        save_button.config(state=NORMAL)
        remove_button.config(state=NORMAL)
    else:
        save_button.config(state=DISABLED)
        remove_button.config(state=DISABLED)

def save_image():
    """
    Сохраняет текущее изображение предпросмотра в файл, выбранный пользователем.
    """
    if img_preview_path:
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg;*.jpeg"),
                                                            ("All files", "*.*")])
        if save_path:
            Image.open(img_preview_path).save(save_path)
            messagebox.showinfo("Информация", f"Изображение сохранено в {save_path}")

def on_checkbox_changed():
    """
    Обрабатывает изменение состояния чекбоксов и применяет соответствующие маски.
    Переключает состояние кнопки удаления в зависимости от выбранных параметров.
    """
    global mask_sound, mask_text, combined_mask_global, original_img

    options = get_selected_options()
    if filepath and options:
        # Применяем маски к оригинальному изображению и обновляем предпросмотр
        thread = threading.Thread(target=apply_masks_and_update_preview, args=(filepath, options))
        thread.start()
    elif filepath and not options:
        # Если не выбран ни один параметр, показываем оригинальное изображение
        update_canvas_image(original_img)
        print("Чекбоксы выключены, показываем оригинальное изображение")
    else:
        print("Нет доступного файла или не выбраны опции")

    # Обновление состояния кнопки удаления
    remove_button.config(state=NORMAL if options and combined_mask_global is not None else DISABLED)



def apply_masks_and_update_preview(filepath, options):
    """
    Применяет маски к изображению и обновляет предпросмотр.
    Обновляет состояние кнопки 'Remove' в зависимости от наличия активных масок.
    """
    global combined_mask_global, img_preview_path
    updated_image_path, combined_mask_global = apply_masks(filepath, options, text_padding, sound_padding, mask_sound, mask_text)
    print(f"apply_masks_and_update_preview: combined_mask_global is None: {combined_mask_global is None}")
    img_preview_path = updated_image_path  # Сохраняем путь к обновленному изображению
    update_preview(updated_image_path)
    unlock_widgets()
    hide_loading_indicator()
    # Обновление состояния кнопки удаления
    remove_button.config(state=NORMAL if any([text_var.get(), sound_var.get()]) and combined_mask_global is not None else DISABLED)


def remove_mask():
    """
    Запускает процесс удаления масок с изображения.
    """
    global filepath, combined_mask_global
    if filepath and combined_mask_global is not None:
        print("Запуск потока для удаления маски")
        lock_widgets()
        show_loading_indicator()
        thread = threading.Thread(target=remove_mask_and_update_preview, args=(filepath, combined_mask_global))
        thread.start()
    else:
        print("Нет доступного файла или маски")

def remove_mask_and_update_preview(filepath, combined_mask):
    """
    Удаляет маски с изображения и обновляет предпросмотр конечного результата.
    """
    global img_preview_path
    # Показываем размытую картинку с нанесенными масками во время работы ламы
    img_with_masks = Image.open(img_preview_path).filter(ImageFilter.GaussianBlur(15))
    update_canvas_image(img_with_masks, is_blurred=True)
    # Запуск ламы для удаления масок
    result_image_path = remove_mask_with_lama(filepath, combined_mask)
    img_preview_path = result_image_path  # Сохраняем путь к результату inpainting
    # Обновляем предпросмотр с конечным результатом
    update_preview(result_image_path)
    unlock_widgets()
    hide_loading_indicator()

# Инициализация окна
window = ttk.Window(themename=current_theme)
window.title("Graphic Novel Annotator")

# Установить размеры окна: ширина 1000 и максимальная высота экрана минус 100 пикселей
screen_height = window.winfo_screenheight() - 100
window.geometry(f"1000x{screen_height}")

# Центрирование окна
window.update_idletasks()
x = (window.winfo_screenwidth() // 2) - (window.winfo_width() // 2)
y = 0
window.geometry(f"+{x}+{y}")

# Создаем рамку для загрузки и сохранения изображений
frame_top = ttk.Frame(window, padding=(5, 5))
frame_top.pack(fill=X)

# Кнопка загрузки изображения
load_button = ttk.Button(frame_top, text="Load image", command=load_photo)
load_button.pack(side=LEFT, padx=2, pady=2)

# Кнопка сохранения изображения
save_button = ttk.Button(frame_top, text="Save", command=save_image, state=DISABLED)
save_button.pack(side=LEFT, padx=2, pady=2)

# Кнопка для смены темы
theme_button = ttk.Button(frame_top, text="🌙", command=switch_theme, width=3)
theme_button.pack(side=RIGHT, padx=2, pady=2)

# Создаем рамку для опций масок
frame_options = ttk.Labelframe(window, text="Options", padding=(5, 5))
frame_options.pack(fill=X, padx=5, pady=5)

# Переменные для чекбоксов опций
text_var = tk.BooleanVar(value=True)
sound_var = tk.BooleanVar(value=True)

# Чекбокс для опции текста
checkbox_text = ttk.Checkbutton(frame_options, text="Text", variable=text_var, command=on_checkbox_changed, bootstyle="success-round-toggle")
checkbox_text.grid(row=0, column=0, padx=2, pady=2)

# Чекбокс для опции звука
checkbox_sound = ttk.Checkbutton(frame_options, text="Sound", variable=sound_var, command=on_checkbox_changed, bootstyle="success-round-toggle")
checkbox_sound.grid(row=0, column=1, padx=2, pady=2)

# Кнопка удаления масок
remove_button = ttk.Button(frame_options, text="Remove", command=remove_mask, state=DISABLED)
remove_button.grid(row=1, column=0, columnspan=2, padx=2, pady=5)

# Создаем рамку для предпросмотра изображения
frame_preview = ttk.Labelframe(window, text="Image Preview", padding=(5, 5))
frame_preview.pack(fill=BOTH, expand=True, padx=5, pady=5)

# Холст для отображения предпросмотра изображения
preview_canvas = tk.Canvas(frame_preview, bg='#325D88')
preview_canvas.pack(fill=BOTH, expand=True)

# Добавляем виджет для индикатора загрузки
progressbar = ttk.Progressbar(preview_canvas, mode='indeterminate', style='info.Horizontal.TProgressbar', length=200)
progressbar.place_forget()


def on_resize(event):
    """
    Обрабатывает изменение размера окна и обновляет изображение на холсте предпросмотра.
    """
    if img_preview_path:
        img = Image.open(img_preview_path)

        # Проверка состояния чекбоксов
        if any([text_var.get(), sound_var.get()]):
            if hasattr(preview_canvas, 'is_blurred') and preview_canvas.is_blurred:
                img = img.filter(ImageFilter.GaussianBlur(15))
            update_canvas_image(img, is_blurred=preview_canvas.is_blurred)
        else:
            # Если ни один чекбокс не активен, показываем оригинальное изображение
            update_canvas_image(original_img)


# Привязка события изменения размера окна
window.bind("<Configure>", on_resize)

# Учет начальных состояний чекбоксов
on_checkbox_changed()

# Инициализация окна и виджетов
def setup_interface():
    update_remove_button_state()  # Устанавливаем начальное состояние кнопки 'Remove'

# Настройка интерфейса и запуск основного цикла окна
setup_interface()
window.mainloop()


