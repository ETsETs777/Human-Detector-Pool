# modules/activity_plot.py
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import threading

class ActivityPlot:
    def __init__(self, parent, human_detector, max_points=100):
        self.parent = parent
        self.human_detector = human_detector
        self.max_points = max_points
        self.data = np.zeros(max_points)
        self.timestamps = [datetime.now() for _ in range(max_points)]
        self.is_active = False
        self.update_thread = None

        self.frame = tk.Frame(parent, bg="#1e1e1e", relief="flat", bd=1) #to do
        self.frame.pack(fill=tk.X, padx=20, pady=(10, 10))

        tk.Label(self.frame, text="📊 Активность человека (в реальном времени)",
                 font=("Segoe UI", 11, "bold"), bg="#1e1e1e", fg="#0078d4").pack(pady=(5, 10))

        self.figure, self.ax = plt.subplots(figsize=(5, 2.5), facecolor="#121212")
        self.ax.set_facecolor("#121212")
        self.ax.spines['bottom'].set_color('#444')
        self.ax.spines['left'].set_color('#444')
        self.ax.tick_params(axis='x', colors='#b0b0b0', labelsize=8)
        self.ax.tick_params(axis='y', colors='#b0b0b0', labelsize=8)
        self.ax.set_ylabel("Обнаружен", color='#ffffff', fontsize=10)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_yticks([0, 1])
        self.ax.set_yticklabels(['Нет', 'Да'], color='#b0b0b0')
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#444')

        self.line, = self.ax.plot([], [], color='#0078d4', linewidth=2)

        self.ax.set_xticks([])

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.X, expand=True)

    def start_update(self):
        if not self.is_active:
            self.is_active = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()

    def stop_update(self):
        self.is_active = False

    def _update_loop(self):
        while self.is_active:
            current_state = 1 if self.human_detector.has_pose_landmarks else 0

            self.data = np.roll(self.data, -1)
            self.data[-1] = current_state

            self.timestamps = self.timestamps[1:] + [datetime.now()]

            self._update_plot()
            self.parent.update_idletasks()
            self.parent.after(1500)

    def _update_plot(self):
        self.line.set_data(range(len(self.data)), self.data)
        self.ax.set_xlim(0, len(self.data))
        self.canvas.draw()