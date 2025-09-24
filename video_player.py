import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("🎥 Проигрыватель видео")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        self.cap = None
        self.is_playing = False
        self.frame_count = 0
        self.fps = 0
        self.total_frames = 0

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(main_frame, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, bg='#1e1e1e')
        control_frame.pack(fill=tk.X, pady=10)

        self.btn_play_pause = tk.Button(control_frame, text="▶️", font=("Segoe UI", 12), bg='#4a90e2', fg='white',
                                        command=self.toggle_play_pause, width=8)
        self.btn_play_pause.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(control_frame, text="⏹️", font=("Segoe UI", 12), bg='#f44336', fg='white',
                                  command=self.stop_video, width=8)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_volume = tk.Button(control_frame, text="🔊", font=("Segoe UI", 12), bg='#2196F3', fg='white',
                                    command=self.toggle_volume, width=8)
        self.btn_volume.pack(side=tk.LEFT, padx=5)

        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(control_frame, textvariable=self.path_var, font=("Segoe UI", 10), width=50,
                                   bg='#2d2d2d', fg='white', insertbackground='white')
        self.path_entry.pack(side=tk.LEFT, padx=10)

        self.btn_open = tk.Button(control_frame, text="🔍 Открыть", font=("Segoe UI", 10), bg='#4CAF50', fg='white',
                                  command=self.open_video_from_path)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_browse = tk.Button(control_frame, text="📁 Выбрать", font=("Segoe UI", 10), bg='#2196F3', fg='white',
                                    command=self.browse_video_file)
        self.btn_browse.pack(side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.progress_var,
                                     bg='#1e1e1e', fg='white',
                                     troughcolor='#333333',
                                     activebackground='#5a9ef2',
                                     length=400, showvalue=False,
                                     command=self.on_progress_change)
        self.progress_bar.pack(side=tk.LEFT, padx=10)

        self.time_label = tk.Label(control_frame, text="00:00 / 00:00", font=("Segoe UI", 10), bg='#1e1e1e', fg='white')
        self.time_label.pack(side=tk.RIGHT, padx=5)

        self.update_time()

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv")]
        )
        if not file_path:
            return
        self.path_var.set(file_path)
        self.load_video(file_path)

    def open_video_from_path(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("Предупреждение", "Введите путь к видео")
            return

        if not os.path.exists(path):
            messagebox.showerror("Ошибка", f"Файл не найден: {path}")
            return

        self.load_video(path)

    def load_video(self, path):
        print(f"Попытка загрузки видео: {path}")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть видео: {path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
        self.is_playing = False
        self.progress_var.set(0)

        self.update_time()
        self.update_frame()

    def update_frame(self):
        if not self.cap or not self.is_playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(700 / w, 500 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        self.frame_count += 1
        progress = (self.frame_count / self.total_frames) * 100
        self.progress_var.set(progress)

        self.update_time()

        delay = int(1000 / self.fps) if self.fps > 0 else 33
        self.root.after(delay, self.update_frame)

    def toggle_play_pause(self):
        if not self.cap:
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.config(text="⏸️")
            self.update_frame()
        else:
            self.btn_play_pause.config(text="▶️")

    def stop_video(self):
        self.is_playing = False
        self.btn_play_pause.config(text="▶️")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")
        self.progress_var.set(0)
        self.frame_count = 0
        self.update_time()

    def toggle_volume(self):
        messagebox.showinfo("Информация", "Звук пока не поддерживается.")

    def on_progress_change(self, value):
        if not self.cap:
            return

        total_seconds = self.total_frames / self.fps
        current_seconds = (float(value) / 100) * total_seconds
        frame_num = int(current_seconds * self.fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        self.frame_count = frame_num
        self.update_frame()

    def update_time(self):
        if not self.cap:
            return

        current_seconds = self.frame_count / self.fps
        total_seconds = self.total_frames / self.fps

        current_str = self.format_time(current_seconds)
        total_str = self.format_time(total_seconds)
        self.time_label.config(text=f"{current_str} / {total_str}")

        self.root.after(100, self.update_time)

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def destroy(self):
        if self.cap:
            self.cap.release()
        self.root.destroy()