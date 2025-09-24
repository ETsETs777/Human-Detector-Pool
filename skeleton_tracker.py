import cv2
import mediapipe as mp
import numpy as np
import os
import time
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import datetime
from PIL import Image, ImageTk
import threading

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LOG_FILE = "logs.txt"

DARK_MODE = True

def get_theme_colors():
    if DARK_MODE:
        return {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#4a90e2',
            'hover': '#5a9ef2',
            'pressed': '#3a80d2',
            'entry_bg': '#2d2d2d',
            'border': '#444444',
            'success': '#4CAF50',
            'error': '#f44336',
            'warning': '#ff9800',
            'progress_bg': '#333333',
            'progress_fg': '#4a90e2',
        }
    else:
        return {
            'bg': '#f5f5f5',
            'fg': '#333333',
            'accent': '#2196F3',
            'hover': '#1e88e5',
            'pressed': '#1976d2',
            'entry_bg': '#ffffff',
            'border': '#cccccc',
            'success': '#43a047',
            'error': '#e53935',
            'warning': '#fb8c00',
            'progress_bg': '#e0e0e0',
            'progress_fg': '#2196F3',
        }

class SkeletonTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 MediaPipe Skeleton Tracker")
        self.root.geometry("1300x900")
        self.root.minsize(900, 650)
        self.colors = get_theme_colors()
        self.setup_styles()
        self.setup_ui()
        self.log_action("Программа запущена")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')

        self.root.configure(bg=self.colors['bg'])

        style.configure('TButton',
                        font=('Segoe UI', 11, 'bold'),
                        padding=10,
                        background=self.colors['accent'],
                        foreground=self.colors['fg'],
                        borderwidth=0)
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
                        lightcolor=self.colors['progress_fg'],
                        darkcolor=self.colors['progress_fg'])

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(main_frame, text="MediaPipe Skeleton Tracker", font=("Segoe UI", 24, "bold"))
        title_label.pack(pady=(0, 30))

        control_frame = ttk.Frame(main_frame, style='TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 20))

        button_frame = ttk.Frame(control_frame, style='TFrame')
        button_frame.pack()

        self.btn_webcam = ttk.Button(button_frame, text="📹 Веб-камера (реальное время)", command=self.start_webcam_thread)
        self.btn_webcam.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_video = ttk.Button(button_frame, text="📂 Обработать видео", command=self.process_video_file)
        self.btn_video.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_logs = ttk.Button(button_frame, text="📜 Показать логи", command=self.show_logs_window)
        self.btn_logs.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_toggle_theme = ttk.Button(button_frame, text="🌓 Сменить тему", command=self.toggle_theme)
        self.btn_toggle_theme.pack(side=tk.LEFT, padx=10, pady=10)

        progress_frame = ttk.Frame(main_frame, style='TFrame')
        progress_frame.pack(fill=tk.X, pady=(0, 20))

        self.progress_label = ttk.Label(progress_frame, text="Готово", font=("Segoe UI", 10))
        self.progress_label.pack(anchor=tk.W)

        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=800)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        log_frame = ttk.LabelFrame(main_frame, text=" 📋 Лог операций ", style='TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame,
                                                  wrap=tk.WORD,
                                                  font=("Consolas", 10),
                                                  bg=self.colors['entry_bg'],
                                                  fg=self.colors['fg'],
                                                  insertbackground=self.colors['fg'],
                                                  relief=tk.FLAT,
                                                  padx=10,
                                                  pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        self.append_log("🚀 Программа готова к работе.\nВыберите действие в меню выше.")

    def toggle_theme(self):
        global DARK_MODE
        DARK_MODE = not DARK_MODE
        self.colors = get_theme_colors()
        self.apply_theme()
        self.log_action(f"Тема переключена на {'тёмную' if DARK_MODE else 'светлую'}")

    def apply_theme(self):
        self.root.configure(bg=self.colors['bg'])

        style = ttk.Style()
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        style.configure('TButton', foreground=self.colors['fg'])
        style.configure('Horizontal.TProgressbar',
                        troughcolor=self.colors['progress_bg'],
                        background=self.colors['progress_fg'])

        self.log_text.config(bg=self.colors['entry_bg'], fg=self.colors['fg'], insertbackground=self.colors['fg'])

        for widget in self.root.winfo_children():
            self._update_widget_colors(widget)

    def _update_widget_colors(self, widget):
        try:
            if isinstance(widget, tk.Label) or isinstance(widget, tk.Button):
                widget.configure(bg=self.colors['bg'], fg=self.colors['fg'])
            elif isinstance(widget, tk.Frame) or isinstance(widget, tk.Toplevel):
                widget.configure(bg=self.colors['bg'])
                for child in widget.winfo_children():
                    self._update_widget_colors(child)
        except:
            pass

    def append_log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        level_tag = f"[{level}]"
        full_message = f"{timestamp} {level_tag} {message}\n"

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, full_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def log_action(self, action):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {action}\n")
        self.append_log(action)

    def start_webcam_thread(self):
        threading.Thread(target=self.process_webcam, daemon=True).start()

    def process_webcam(self):
        self.log_action("Запущена обработка в реальном времени (веб-камера)")
        self.update_progress("Подключение к веб-камере...", 0)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            error_msg = "❌ Не удалось открыть веб-камеру."
            self.append_log(error_msg, "ERROR")
            self.log_action("Ошибка: Не удалось открыть веб-камеру")
            self.update_progress("Ошибка", 0)
            messagebox.showerror("Ошибка", error_msg)
            return

        self.update_progress("Запущена веб-камера. Нажмите 'q' для выхода.", 100)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                self.human_detector.update(
                    has_pose_landmarks=bool(results.pose_landmarks),
                    context="видео",
                    frame_num=frame_num,
                    current_frame=image
                )
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                cv2.imshow('MediaPipe Skeleton (q - выход)', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.update_progress("Веб-камера отключена", 0)
        self.log_action("Завершена обработка в реальном времени")
        self.append_log("✅ Веб-камера закрыта.")

    def process_video_file(self):
        video_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")]
        )

        if not video_path:
            self.append_log("❌ Файл не выбран.", "WARNING")
            self.log_action("Попытка обработки видео: файл не выбран")
            return

        save_dir = filedialog.askdirectory(title="Выберите папку для сохранения обработанного видео")
        if not save_dir:
            self.append_log("❌ Папка не выбрана. Сохранение отменено.", "WARNING")
            self.log_action("Попытка обработки видео: папка сохранения не выбрана")
            return

        save_path = os.path.join(save_dir, "processed_" + os.path.basename(video_path))
        self.log_action(f"Начата обработка видео: {video_path} → {save_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"❌ Не удалось открыть видеофайл: {video_path}"
            self.append_log(error_msg, "ERROR")
            self.log_action(f"Ошибка: не удалось открыть видеофайл {video_path}")
            messagebox.showerror("Ошибка", error_msg)
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            error_msg = f"❌ Не удалось создать выходной файл: {save_path}"
            self.append_log(error_msg, "ERROR")
            self.log_action(f"Ошибка: не удалось создать выходной файл {save_path}")
            cap.release()
            messagebox.showerror("Ошибка", error_msg)
            return

        info_msg = f"🎥 Обработка: {os.path.basename(video_path)} | 🖼️ {total_frames} кадров"
        self.append_log(info_msg)
        self.update_progress("Обработка начата...", 0)

        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

            for frame_num in range(total_frames):
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                out.write(image)

                progress = int((frame_num + 1) / total_frames * 100)
                self.update_progress(f"Обработка... {progress}%", progress)

        cap.release()
        out.release()

        success_msg = f"✅ Видео сохранено: {save_path}"
        self.append_log(success_msg, "SUCCESS")
        self.log_action(f"Видео обработано и сохранено: {save_path}")
        self.update_progress("Готово", 100)
        messagebox.showinfo("Успешно", success_msg)

    def update_progress(self, text, value):
        self.progress_label.config(text=text)
        self.progress_bar['value'] = value
        self.root.update_idletasks()

    def show_logs_window(self):
        log_window = tk.Toplevel(self.root)
        log_window.title("📜 Полная история операций")
        log_window.geometry("800x600")
        log_window.configure(bg=self.colors['bg'])

        ttk.Label(log_window, text="История операций", font=("Segoe UI", 16, "bold")).pack(pady=10)

        # Текстовое поле
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

        ttk.Button(log_window, text="Закрыть", command=log_window.destroy).pack(pady=10) #to do

def main():
    root = tk.Tk()
    app = SkeletonTrackerApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()