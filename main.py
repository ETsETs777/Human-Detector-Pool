import cv2
import mediapipe as mp
import numpy as np
import os
import time
import json
import sys
import threading
import traceback
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import datetime
from PIL import Image, ImageTk
import math

# 👇 Импорты для PRO-функций
from modules.activity_plot import ActivityPlot
from modules.data_exporter import DataExporter
from modules.human_detector import HumanDetector
from i18n import t

# 👇 Подавляем предупреждения TensorFlow и OpenCV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LOG_FILE = "logs.txt"
ERROR_LOG = "errors.log"
DARK_MODE = True

def get_theme_colors():
    return {
        'bg': '#121212',
        'fg': '#ffffff',
        'accent': '#0078d4',
        'hover': '#0096ff',
        'pressed': '#005a9e',
        'entry_bg': '#1e1e1e',
        'border': '#2c2c2c',
        'success': '#21b35f',
        'error': '#e74c3c',
        'warning': '#f39c12',
        'progress_bg': '#2d2d2d',
        'progress_fg': '#0078d4',
        'status_bg': '#1a1a1a',
        'status_fg': '#b0b0b0',
        'card_bg': '#252525',
    }

def get_light_theme_colors():
    return {
        'bg': '#f5f5f5',
        'fg': '#333333',
        'accent': '#0078d4',
        'hover': '#0096ff',
        'pressed': '#005a9e',
        'entry_bg': '#ffffff',
        'border': '#cccccc',
        'success': '#21b35f',
        'error': '#e74c3c',
        'warning': '#f39c12',
        'progress_bg': '#e0e0e0',
        'progress_fg': '#0078d4',
        'status_bg': '#e8e8e8',
        'status_fg': '#666666',
        'card_bg': '#ffffff',
    }

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", 10))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
        self.tip_window = None

class SkeletonTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(t("title"))
        self.root.state('zoomed')
        self.root.minsize(900, 650)

        # 🔒 Защита от двойного запуска
        if self.is_already_running():
            self.show_duplicate_warning()
            sys.exit(0)

        self.colors = get_theme_colors()
        self.particle_bg = None  # Будет создано позже
        self.setup_styles()
        self.icons = self.load_icons()
        self.settings_file = "settings.json"
        self.language = "ru"
        self.load_settings()

        # 🟢 Создаём детектор человека ПЕРЕД UI
        self.screenshot_dir = "screenshots"
        self.exports_dir = "exports"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)

        self.human_detector = HumanDetector(
            log_callback=self.append_log,
            log_action_callback=self.log_action,
            screenshot_dir=self.screenshot_dir
        )

        self.data_exporter = DataExporter(self.human_detector)

        self.is_camera_active = False
        self.last_raw_frame = None
        self.last_processed_frame = None
        self.flask_thread = None
        self.is_flask_running = False

        # 🟡 ЗАПУСКАЕМ UI
        self.setup_ui()

        # 🟢 Сохраняем ссылки на кнопки
        self.btn_webcam = None
        self.btn_screenshot = None
        self.btn_stop_camera = None

        # 🟢 Привязываем горячие клавиши
        self.root.bind('<F1>', lambda e: self.start_webcam_thread())
        self.root.bind('<F2>', lambda e: self.process_video_file())
        self.root.bind('<F3>', lambda e: self.show_logs_window())
        self.root.bind('<F4>', lambda e: self.toggle_background())
        self.root.bind('<F5>', lambda e: self.open_video_player())
        self.root.bind('<Escape>', lambda e: self.stop_camera())
        self.root.bind('B', lambda e: self.show_error_log())
        self.root.bind('A', lambda e: self.start_flask_server())
        self.root.bind('L', lambda e: self.select_language())

        # Автозапуск камеры по настройкам
        if self.settings.get("auto_start_camera", False):
            self.start_webcam_thread()

        self.log_action("Программа запущена")

    def is_already_running(self):
        import psutil
        current_process = psutil.Process()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python.exe' and proc.info['pid'] != current_process.pid:
                cmdline = proc.info['cmdline']
                if cmdline and any('main.py' in arg for arg in cmdline):
                    return True
        return False

    def show_duplicate_warning(self):
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(t("already_running"), t("already_running"))
        root.destroy()

    def load_settings(self):
        default_settings = {
            "theme": "dark",
            "auto_start_camera": False,
            "language": "ru",
            "autoscreenshot_threshold": 3.0
        }

        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings = json.load(f)
            except Exception as e:
                self.append_log(f"⚠️ Ошибка загрузки settings.json: {e}", "ERROR")
                self.settings = default_settings
        else:
            self.settings = default_settings
            self.save_settings()

        # Применяем тему
        if self.settings["theme"] == "light":
            self.colors = get_light_theme_colors()
        else:
            self.colors = get_theme_colors()

        # Применяем язык
        self.language = self.settings["language"]
        self.t = lambda key: t(key, self.language)

    def save_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.append_log(f"❌ Не удалось сохранить настройки: {e}", "ERROR")

    def load_icons(self):
        icons = {}
        icon_names = [
            'webcam', 'video', 'logs', 'bg', 'player', 'screenshot',
            'stop', 'graph', 'export', 'settings', 'bug', 'api',
            'check_camera', 'save_settings', 'language'
        ]
        icon_dir = "icons"
        if not os.path.exists(icon_dir):
            return icons
        for name in icon_names:
            path = os.path.join(icon_dir, f"{name}.png")
            if os.path.exists(path):
                img = Image.open(path).resize((20, 20), Image.LANCZOS)
                icons[name] = ImageTk.PhotoImage(img)
        return icons

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        self.root.configure(bg=self.colors['bg'])
        style.configure('TButton',
                        font=('Segoe UI', 11, 'bold'),
                        padding=10,
                        background=self.colors['accent'],
                        foreground=self.colors['fg'],
                        borderwidth=0,
                        relief='flat')
        style.map('TButton',
                  background=[('active', self.colors['hover']), ('pressed', self.colors['pressed'])])
        style.configure('TLabel',
                        font=('Segoe UI', 11),
                        background=self.colors['bg'],
                        foreground=self.colors['fg'])
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Horizontal.TProgressbar',
                        troughcolor=self.colors['progress_bg'],
                        background=self.colors['progress_fg'],
                        bordercolor=self.colors['border'],
                        thickness=12,
                        relief='flat')

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        # 📈 Компактный график активности
        self.activity_plot = ActivityPlot(main_frame, self.human_detector)

        title_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        title_frame.pack(pady=(0, 30))

        title_label = tk.Label(title_frame,
                               text=self.t("title"),
                               font=("Segoe UI", 22, "bold"),
                               fg=self.colors['fg'],
                               bg=self.colors['bg'],
                               padx=20,
                               pady=10)
        title_label.pack()

        def animate_title_color():
            colors = [self.colors['accent'], self.colors['hover'], self.colors['success'], self.colors['warning'], self.colors['error']]
            def cycle(index=0):
                title_label.config(fg=colors[index])
                self.root.after(800, cycle, (index + 1) % len(colors))
            cycle()
        animate_title_color()

        control_frame = ttk.Frame(main_frame, style='TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 30))

        groups = [
            (self.t("camera"), [
                (self.t("start_camera"), self.start_webcam_thread, 'webcam', '<F1>'),
                (self.t("screenshot"), self.take_screenshot, 'screenshot', 'S'),
                (self.t("stop"), self.stop_camera, 'stop', '<Esc>')
            ], self.colors['accent']),
            (self.t("video"), [
                (self.t("process_video"), self.process_video_file, 'video', '<F2>'),
                (self.t("player"), self.open_video_player, 'player', '<F5>')
            ], self.colors['success']),
            (self.t("services"), [
                (self.t("logs"), self.show_logs_window, 'logs', '<F3>'),
                (self.t("background"), self.toggle_background, 'bg', '<F4>'),
                (self.t("graph"), self.toggle_activity_plot, 'graph', 'G'),
                (self.t("export"), self.export_data, 'export', 'E'),
                (self.t("settings"), self.open_settings, 'settings', 'P'),
                (self.t("error_log"), self.show_error_log, 'bug', 'B'),
                (self.t("flask_api"), self.start_flask_server, 'api', 'A'),
                (self.t("check_camera"), self.check_camera, 'check_camera', 'C'),
                (self.t("save_settings"), self.save_settings_now, 'save_settings', 'S'),
                (self.t("select_language"), self.select_language, 'language', 'L')
            ], self.colors['warning'])
        ]

        for group_title, buttons, color in groups:
            group_frame = tk.Frame(control_frame, bg=self.colors['bg'], highlightbackground=color,
                                   highlightthickness=2, bd=0, relief="flat")
            group_frame.pack(side=tk.LEFT, padx=15, fill=tk.Y, expand=True)

            group_label = tk.Label(group_frame, text=group_title, font=("Segoe UI", 12, "bold"),
                                   bg=self.colors['bg'], fg=color)
            group_label.pack(pady=(10, 15))

            for text, command, icon_key, tooltip_key in buttons:
                btn = tk.Button(group_frame,
                                text=text,
                                image=self.icons.get(icon_key),
                                compound=tk.LEFT,
                                font=("Segoe UI", 10, "bold"),
                                bg=self.colors['entry_bg'],
                                fg=self.colors['fg'],
                                activebackground=self.colors['hover'],
                                activeforeground=self.colors['fg'],
                                relief="flat",
                                bd=0,
                                padx=15,
                                pady=8,
                                cursor="hand2",
                                command=command)
                btn.pack(padx=10, pady=5, fill=tk.X)
                btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.colors['hover']))
                btn.bind("<Leave>", lambda e, b=btn: b.config(bg=self.colors['entry_bg']))
                hotkey = tooltip_key.replace('<', '').replace('>', '')
                ToolTip(btn, f"{hotkey}: {text}")

                clean_text = text.strip()
                if clean_text == self.t("start_camera"):
                    self.btn_webcam = btn
                elif clean_text == self.t("screenshot"):
                    self.btn_screenshot = btn
                elif clean_text == self.t("stop"):
                    self.btn_stop_camera = btn

        progress_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        progress_frame.pack(fill=tk.X, pady=(0, 25))

        self.progress_label = tk.Label(progress_frame, text=self.t("ready"), font=("Segoe UI", 10, "italic"),
                                       bg=self.colors['bg'], fg=self.colors['status_fg'])
        self.progress_label.pack(anchor=tk.W)

        self.progress_canvas = tk.Canvas(progress_frame, height=20, bg=self.colors['progress_bg'],
                                         highlightthickness=0, relief='flat')
        self.progress_canvas.pack(fill=tk.X, pady=(5, 0))
        self.progress_rect = self.progress_canvas.create_rectangle(0, 0, 0, 20, fill=self.colors['progress_fg'], outline="")
        self.progress_text = self.progress_canvas.create_text(400, 10, text="0%", fill=self.colors['fg'], font=("Segoe UI", 9))

        log_frame = tk.LabelFrame(main_frame, text=f" 📋 {self.t('logs')} ", font=("Segoe UI", 12, "bold"),
                                  bg=self.colors['bg'], fg=self.colors['accent'], relief="flat",
                                  highlightbackground=self.colors['border'], highlightthickness=1)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        self.log_text = tk.Text(log_frame,
                                wrap=tk.WORD,
                                font=("Consolas", 10),
                                bg=self.colors['entry_bg'],
                                fg=self.colors['fg'],
                                insertbackground=self.colors['fg'],
                                relief=tk.FLAT,
                                padx=15,
                                pady=15,
                                height=10,
                                spacing1=3,
                                spacing3=5)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        self.log_text.tag_config("INFO", foreground=self.colors['fg'], lmargin1=10, lmargin2=10)
        self.log_text.tag_config("SUCCESS", foreground=self.colors['success'], lmargin1=10, lmargin2=10)
        self.log_text.tag_config("ERROR", foreground=self.colors['error'], lmargin1=10, lmargin2=10)
        self.log_text.tag_config("WARNING", foreground=self.colors['warning'], lmargin1=10, lmargin2=10)

        self.status_bar = tk.Label(self.root, text=self.t("ready"),
                                   bg=self.colors['status_bg'],
                                   fg=self.colors['status_fg'],
                                   font=("Segoe UI", 10),
                                   anchor=tk.W,
                                   padx=20, pady=8)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.append_log(self.t("started"), "INFO")

    def _update_button_states(self):
        if self.is_camera_active:
            if hasattr(self, 'btn_webcam') and self.btn_webcam is not None:
                self.btn_webcam.config(state=tk.DISABLED)
            if hasattr(self, 'btn_screenshot') and self.btn_screenshot is not None:
                self.btn_screenshot.config(state=tk.NORMAL)
            if hasattr(self, 'btn_stop_camera') and self.btn_stop_camera is not None:
                self.btn_stop_camera.config(state=tk.NORMAL)
        else:
            if hasattr(self, 'btn_webcam') and self.btn_webcam is not None:
                self.btn_webcam.config(state=tk.NORMAL)
            if hasattr(self, 'btn_screenshot') and self.btn_screenshot is not None:
                self.btn_screenshot.config(state=tk.DISABLED)
            if hasattr(self, 'btn_stop_camera') and self.btn_stop_camera is not None:
                self.btn_stop_camera.config(state=tk.DISABLED)

    def append_log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"{timestamp} [{level}] {message}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full_message, level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_bar.config(text=f"{level}: {message}")

    def log_action(self, action):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {action}\n")

    def start_webcam_thread(self):
        if self.is_camera_active:
            self.append_log(self.t("camera_off"), "WARNING")
            return
        threading.Thread(target=self.process_webcam, daemon=True).start()

    def process_webcam(self):
        self.log_action("Запущена обработка в реальном времени (веб-камера)")
        self.update_progress(self.t("loading"), 0)

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                error_msg = self.t("camera_off")
                self.append_log(error_msg, "ERROR")
                self.log_action("Ошибка: Не удалось открыть веб-камеру")
                self.update_progress(error_msg, 0)
                messagebox.showerror("Ошибка", error_msg)
                with open(ERROR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now()}] ERROR: Не удалось открыть веб-камеру\n")
                    f.write(traceback.format_exc() + "\n\n")
                self.root.after(0, self._update_button_states)
                return
        except Exception as e:
            error_msg = self.t("camera_off")
            self.append_log(f"{error_msg}: {str(e)}", "ERROR")
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now()}] ERROR: {error_msg}\n")
                f.write(traceback.format_exc() + "\n\n")
            messagebox.showerror("Ошибка", error_msg)
            self.root.after(0, self._update_button_states)
            return

        self.is_camera_active = True
        self.root.after(0, self._update_button_states)
        self.update_progress(self.t("started"), 100)

        self.human_detector.reset()
        self.activity_plot.start_update()

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            while self.is_camera_active and cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                self.last_raw_frame = image.copy()

                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                image.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                self.human_detector.update(
                    has_pose_landmarks=bool(results.pose_landmarks),
                    context="веб-камера",
                    current_frame=image_bgr
                )

                self.last_processed_frame = image_bgr.copy()

                cv2.imshow('MediaPipe Skeleton (q - выход, s - скриншот)', image_bgr)

                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.root.after(0, self.take_screenshot)

        cap.release()
        cv2.destroyAllWindows()
        self.is_camera_active = False
        self.root.after(0, self._update_button_states)
        self.update_progress(self.t("ready"), 0)
        self.append_log("✅ Веб-камера закрыта.", "SUCCESS")
        self.activity_plot.stop_update()

    def stop_camera(self, event=None):
        if self.is_camera_active:
            self.is_camera_active = False
            self.append_log("⏹️ Камера остановлена по запросу пользователя.", "INFO")
            self.log_action("Камера остановлена вручную")
        else:
            self.append_log("⚠️ Камера не активна.", "WARNING")

    def take_screenshot(self):
        if self.last_processed_frame is None and self.last_raw_frame is None:
            self.append_log("Нет доступного кадра для скриншота.", "ERROR")
            messagebox.showwarning("Предупреждение", "Нет активного изображения для сохранения.")
            return

        save_both = messagebox.askyesno("Скриншот",
                                        "Сохранить оба варианта (сырой + с наложением)?\nЕсли 'Нет' — будет запрошен выбор.")

        if save_both:
            frames = [("сырой", self.last_raw_frame), ("с наложением", self.last_processed_frame)]
        else:
            use_processed = messagebox.askyesno("Скриншот", "Сохранить кадр с наложенным скелетом?")
            frame = self.last_processed_frame if use_processed else self.last_raw_frame
            desc = "с наложением" if use_processed else "сырой"
            frames = [(desc, frame)]

        for desc, frame in frames:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{desc}_{timestamp}.jpg"
            filepath = os.path.join(self.screenshot_dir, filename)

            success = cv2.imwrite(filepath, frame)
            if success:
                message = f"📸 {self.t('screenshot_saved')} {filepath}"
                self.append_log(message, "SUCCESS")
                self.log_action(f"Скриншот сохранён: {filename}")
            else:
                self.append_log(f"Не удалось сохранить скриншот {desc}.", "ERROR")

        if save_both:
            messagebox.showinfo("Скриншот", f"✅ Оба скриншота сохранены в папку:\n{self.screenshot_dir}")
        else:
            messagebox.showinfo("Скриншот", "✅ Скриншот сохранён.")

    def process_video_file(self):
        video_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")]
        )
        if not video_path:
            self.append_log("Файл не выбран.", "WARNING")
            return

        save_dir = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not save_dir:
            self.append_log("Папка не выбрана. Сохранение отменено.", "WARNING")
            return

        save_path = os.path.join(save_dir, "processed_" + os.path.basename(video_path))
        self.log_action(f"Начата обработка видео: {video_path} → {save_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"Не удалось открыть видеофайл: {video_path}"
            self.append_log(error_msg, "ERROR")
            messagebox.showerror("Ошибка", f"❌ Не удалось открыть видеофайл:\n{video_path}")
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now()}] ERROR: {error_msg}\n")
                f.write(traceback.format_exc() + "\n\n")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            error_msg = f"Не удалось создать выходной файл: {save_path}"
            self.append_log(error_msg, "ERROR")
            cap.release()
            messagebox.showerror("Ошибка", f"❌ Не удалось создать файл:\n{save_path}")
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.datetime.now()}] ERROR: {error_msg}\n")
                f.write(traceback.format_exc() + "\n\n")
            return

        self.append_log(f"Обработка: {os.path.basename(video_path)} | {total_frames} кадров", "INFO")
        self.update_progress("Обработка начата...", 0)

        self.human_detector.reset()

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            for frame_num in range(total_frames):
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                image.flags.writeable = True
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                self.human_detector.update(
                    has_pose_landmarks=bool(results.pose_landmarks),
                    context="видео",
                    frame_num=frame_num,
                    current_frame=image
                )

                out.write(image)

                progress = int((frame_num + 1) / total_frames * 100)
                self.update_progress(f"Обработка... {progress}%", progress)

        cap.release()
        out.release()

        self.append_log(f"{self.t('video_saved')} {save_path}", "SUCCESS")
        self.update_progress("Готово", 100)
        messagebox.showinfo("Успешно", f"✅ {self.t('video_saved')}\n{save_path}")
        self.export_data()

    def update_progress(self, text, value):
        self.progress_label.config(text=text)
        canvas_width = self.progress_canvas.winfo_width() or 800
        fill_width = int((value / 100) * canvas_width)
        self.progress_canvas.coords(self.progress_rect, 0, 0, fill_width, 20)
        self.progress_canvas.itemconfig(self.progress_text, text=f"{value}%" if value > 0 else "")
        self.root.update_idletasks()

    def show_logs_window(self):
        log_window = tk.Toplevel(self.root)
        log_window.title("📜 Полная история операций")
        log_window.geometry("800x600")
        log_window.configure(bg=self.colors['bg'])

        ttk.Label(log_window, text="История операций", font=("Segoe UI", 16, "bold")).pack(pady=10)

        log_text_full = scrolledtext.ScrolledText(log_window,
                                                  wrap=tk.WORD,
                                                  font=("Consolas", 10),
                                                  bg=self.colors['entry_bg'],
                                                  fg=self.colors['fg'],
                                                  insertbackground=self.colors['fg'],
                                                  relief=tk.FLAT,
                                                  padx=10,
                                                  pady=10)
        log_text_full.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                log_text_full.insert(tk.END, content)
        else:
            log_text_full.insert(tk.END, "Лог-файл отсутствует.")

        log_text_full.config(state=tk.DISABLED)
        ttk.Button(log_window, text="Закрыть", command=log_window.destroy).pack(pady=10)

    def toggle_background(self):
        is_enabled = self.particle_bg.toggle()
        status = "включён" if is_enabled else "выключен"
        self.log_action(f"Анимированный фон {status}")
        self.append_log(f"🌌 Анимированный фон {status}", "INFO")

    def open_video_player(self):
        player_window = tk.Toplevel(self.root)
        player_window.title("🎥 Проигрыватель видео")
        player_window.geometry("800x600")
        player_window.minsize(600, 400)

        player = VideoPlayer(player_window)
        self.log_action("Открыт проигрыватель видео")
        self.append_log("▶️ Открыт проигрыватель видео", "INFO")

    def toggle_activity_plot(self):
        if hasattr(self, 'activity_plot'):
            if self.activity_plot.is_active:
                self.activity_plot.stop_update()
                self.append_log("⏸️ График остановлен.", "INFO")
            else:
                self.activity_plot.start_update()
                self.append_log("▶️ График запущен.", "INFO")

    def export_data(self):
        success, message = self.data_exporter.export_all()
        if success:
            self.append_log(message, "SUCCESS")
            messagebox.showinfo("Экспорт", message)
        else:
            self.append_log(message, "ERROR")
            messagebox.showwarning("Ошибка", message)

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title(self.t("settings_title"))
        settings_win.geometry("400x300")
        settings_win.configure(bg=self.colors['bg'])
        settings_win.resizable(False, False)

        tk.Label(settings_win, text=self.t("settings_title"), font=("Segoe UI", 14, "bold"),
                 bg=self.colors['bg'], fg=self.colors['accent']).pack(pady=(10, 20))
        tk.Label(settings_win, text="Тема:", font=("Segoe UI", 10), bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor='w', padx=20)
        theme_var = tk.StringVar(value=self.settings["theme"])
        tk.Radiobutton(settings_win, text=self.t("theme_dark"), variable=theme_var, value="dark",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg']).pack(anchor='w', padx=30)
        tk.Radiobutton(settings_win, text=self.t("theme_light"), variable=theme_var, value="light",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg']).pack(anchor='w', padx=30)
        tk.Label(settings_win, text="Язык:", font=("Segoe UI", 10), bg=self.colors['bg'], fg=self.colors['fg']).pack(anchor='w', padx=20, pady=(10,0))
        lang_var = tk.StringVar(value=self.settings["language"])
        tk.Radiobutton(settings_win, text=self.t("lang_ru"), variable=lang_var, value="ru",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg']).pack(anchor='w', padx=30)
        tk.Radiobutton(settings_win, text=self.t("lang_en"), variable=lang_var, value="en",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg']).pack(anchor='w', padx=30)
        auto_start_var = tk.BooleanVar(value=self.settings["auto_start_camera"])
        tk.Checkbutton(settings_win, text=self.t("auto_start"),
                       variable=auto_start_var, bg=self.colors['bg'], fg=self.colors['fg'],
                       selectcolor=self.colors['entry_bg']).pack(anchor='w', padx=20, pady=(10,5))

        # Сохранить
        def save_and_close():
            self.settings["theme"] = theme_var.get()
            self.settings["language"] = lang_var.get()
            self.settings["auto_start_camera"] = auto_start_var.get()
            self.save_settings()
            if self.settings["theme"] == "light":
                self.colors = get_light_theme_colors()
            else:
                self.colors = get_theme_colors()
            self.setup_styles()
            self.language = self.settings["language"]
            self.t = lambda key: t(key, self.language)
            self.reload_ui_texts()
            settings_win.destroy()

        tk.Button(settings_win, text="💾 Сохранить", command=save_and_close,
                  bg=self.colors['accent'], fg="white", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=20, pady=8).pack(pady=20)

    def show_error_log(self):
        if not os.path.exists(ERROR_LOG):
            messagebox.showinfo("Ошибки", "Файл ошибок не найден.")
            return

        log_win = tk.Toplevel(self.root)
        log_win.title(self.t("error_log_title"))
        log_win.geometry("800x500")
        log_win.configure(bg=self.colors['bg'])

        text = scrolledtext.ScrolledText(log_win, wrap=tk.WORD, font=("Consolas", 9),
                                         bg=self.colors['entry_bg'], fg=self.colors['fg'],
                                         relief=tk.FLAT, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        with open(ERROR_LOG, "r", encoding="utf-8") as f:
            content = f.read()
            text.insert(tk.END, content)
        text.config(state=tk.DISABLED)

    def start_flask_server(self):
        from flask import Flask, jsonify
        import threading

        if self.is_flask_running:
            messagebox.showinfo("API", "Сервер уже запущен.\nПерейти: http://127.0.0.1:5000/status")
            return

        app = Flask(__name__)

        @app.route('/status')
        def status():
            return jsonify({
                "active": self.is_camera_active,
                "person_detected": self.human_detector.has_pose_landmarks,
                "fps": 0,
                "detections": len(self.human_detector.detection_history),
                "last_seen": self.human_detector.last_detection_time
            })

        @app.route('/shutdown', methods=['POST'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func:
                func()
            return "Server shutting down..."

        def run_app():
            app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)

        self.flask_thread = threading.Thread(target=run_app, daemon=True)
        self.flask_thread.start()
        self.is_flask_running = True
        self.append_log("🌐 Flask API запущен на http://127.0.0.1:5000/status", "SUCCESS")
        messagebox.showinfo("API", "Сервер запущен!\nПерейдите в браузер:\nhttp://127.0.0.1:5000/status")

    def check_camera(self):
        available = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available.append(f"Камера {i}")
                cap.release()

        if available:
            message = "Доступные камеры:\n\n" + "\n".join(available)
            messagebox.showinfo(self.t("status"), message)
        else:
            messagebox.showwarning(self.t("status"), "Не найдено ни одной рабочей камеры.")

    def save_settings_now(self):
        self.save_settings()
        messagebox.showinfo(self.t("settings"), self.t("export_success"))

    def select_language(self):
        lang_dialog = tk.Toplevel(self.root)
        lang_dialog.title(self.t("language"))
        lang_dialog.geometry("300x250")
        lang_dialog.configure(bg=self.colors['bg'])
        lang_dialog.resizable(False, False)
        lang_dialog.transient(self.root)
        lang_dialog.grab_set()

        tk.Label(lang_dialog, text=self.t("select_language"), font=("Segoe UI", 12, "bold"),
                 bg=self.colors['bg'], fg=self.colors['accent']).pack(pady=(20, 10))
        current_lang = self.language
        selected_lang = tk.StringVar(value=current_lang)

        tk.Radiobutton(lang_dialog, text=self.t("lang_ru"), variable=selected_lang, value="ru",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg'],
                       font=("Segoe UI", 10)).pack(anchor='w', padx=40, pady=5)
        tk.Radiobutton(lang_dialog, text=self.t("lang_en"), variable=selected_lang, value="en",
                       bg=self.colors['bg'], fg=self.colors['fg'], selectcolor=self.colors['entry_bg'],
                       font=("Segoe UI", 10)).pack(anchor='w', padx=40, pady=5)

        def apply_language():
            new_lang = selected_lang.get()
            if new_lang != self.language:
                self.language = new_lang
                self.settings["language"] = new_lang
                self.save_settings()
                self.t = lambda key: t(key, self.language)
                self.setup_styles()
                self.reload_ui_texts()
                lang_dialog.destroy()
                messagebox.showinfo(self.t("settings"), f"{self.t('language')} {self.t(f'lang_{new_lang}')}")

        btn_apply = tk.Button(lang_dialog, text="✓ Применить", command=apply_language,
                              bg=self.colors['accent'], fg="white", font=("Segoe UI", 10, "bold"),
                              relief="flat", padx=20, pady=82)
        btn_apply.pack(pady=20)
        btn_cancel = tk.Button(lang_dialog, text="✖ Отмена", command=lang_dialog.destroy,
                               bg=self.colors['entry_bg'], fg=self.colors['fg'], font=("Segoe UI", 10),
                               relief="flat", padx=20, pady=8)
        btn_cancel.pack(pady=(0, 20))

    def reload_ui_texts(self):
        self.root.title(self.t("title"))

        for child in self.control_frame.winfo_children():
            if isinstance(child, tk.Frame) and len(child.winfo_children()) > 0:
                first_label = child.winfo_children()[0]
                if isinstance(first_label, tk.Label) and first_label.cget("text") in ["📹 Камера", "📂 Видео", "⚙️ Сервисы"]:
                    group_text = first_label.cget("text")
                    if group_text == "📹 Камера":
                        first_label.config(text=self.t("camera"))
                    elif group_text == "📂 Видео":
                        first_label.config(text=self.t("video"))
                    elif group_text == "⚙️ Сервисы":
                        first_label.config(text=self.t("services"))

        for group_frame in self.control_frame.winfo_children():
            if isinstance(group_frame, tk.Frame):
                for btn in group_frame.winfo_children():
                    if isinstance(btn, tk.Button):
                        current_text = btn.cget("text")
                        for key, translated in self._reverse_translate_map().items():
                            if current_text == translated:
                                btn.config(text=self.t(key))
                                break

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            self.log_text.insert(tk.END, content)
        self.log_text.config(state=tk.DISABLED)

        self.progress_label.config(text=self.t("ready"))
        self.status_bar.config(text=self.t("ready"))

        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel) and widget.title() == "📜 Полная история операций":
                widget.title("📜 " + self.t("logs"))

    def _reverse_translate_map(self):
        reverse_map = {}
        for lang_dict in translations.values():
            for key, value in lang_dict.items():
                reverse_map[key] = value
        return reverse_map

    def on_closing(self):
        if self.is_flask_running:
            import requests
            try:
                requests.post('http://127.0.0.1:5000/shutdown')
            except:
                pass
        if self.particle_bg:
            self.particle_bg.destroy()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SkeletonTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()