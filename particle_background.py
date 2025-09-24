import tkinter as tk
import math
import numpy as np

class Particle:
    def __init__(self, x, y, vx, vy, canvas_width, canvas_height):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.radius = 1.2

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x <= 0 or self.x >= self.canvas_width:
            self.vx *= -1
        if self.y <= 0 or self.y >= self.canvas_height:
            self.vy *= -1

    def draw(self, canvas, color):
        canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill=color, outline="", tags="particle"
        )

class ParticleBackground:
    def __init__(self, root):
        self.root = root
        self.canvas = None
        self.particles = []
        self.animation_running = True
        self._resize_timer = None

        self.create_canvas()
        self.init_particles()
        self.start_animation()

    def create_canvas(self):
        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='black')
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)

        self.root.bind('<Configure>', lambda e: self.schedule_init_particles())

    def schedule_init_particles(self):
        if self._resize_timer:
            self.root.after_cancel(self._resize_timer)
        self._resize_timer = self.root.after(200, self.init_particles)

    def init_particles(self, event=None):
        if not self.canvas or not self.canvas.winfo_exists():
            return
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width <= 1 or height <= 1:
            return

        self.particles.clear()
        for _ in range(80):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            vx = (np.random.rand() - 0.5) * 1.5
            vy = (np.random.rand() - 0.5) * 1.5
            self.particles.append(Particle(x, y, vx, vy, width, height))

    def animate(self):
        if not self.animation_running or not self.canvas or not self.canvas.winfo_exists():
            return

        width = self.root.winfo_width()
        height = self.root.winfo_height()
        if width <= 1 or height <= 1:
            self.root.after(50, self.animate)
            return

        for p in self.particles:
            p.update()

        self.canvas.delete("all")

        particle_color = "#ffffff"
        line_base = (100, 150, 255)

        for i, p1 in enumerate(self.particles):
            for p2 in self.particles[i+1:]:
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 120:
                    alpha = max(0, (120 - dist) / 120)
                    r = int(line_base[0] * alpha)
                    g = int(line_base[1] * alpha)
                    b = int(line_base[2] * alpha)
                    line_color = f"#{r:02x}{g:02x}{b:02x}"
                    self.canvas.create_line(
                        p1.x, p1.y, p2.x, p2.y,
                        fill=line_color, width=0.5, tags="line"
                    )

        for p in self.particles:
            p.draw(self.canvas, particle_color)

        self.root.after(40, self.animate)

    def start_animation(self):
        self.animation_running = True
        self.animate()

    def stop_animation(self):
        self.animation_running = False

    def toggle(self):
        if self.canvas and self.canvas.winfo_exists():
            self.stop_animation()
            self.canvas.destroy()
            self.canvas = None
            return False
        else:
            self.create_canvas()
            self.init_particles()
            self.start_animation()
            return True

    def destroy(self):
        self.stop_animation()
        if self.canvas and self.canvas.winfo_exists():
            self.canvas.destroy()
        self.canvas = None