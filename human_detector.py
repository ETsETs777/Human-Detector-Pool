import cv2
import os
from datetime import datetime
import time

class HumanDetector:
    def __init__(self, log_callback=None, log_action_callback=None, screenshot_dir="screenshots", autoscreenshot=False):
        self.log_callback = log_callback
        self.log_action_callback = log_action_callback
        self.screenshot_dir = screenshot_dir
        self.autoscreenshot = autoscreenshot
        self.is_detected = False
        self.detection_history = []
        self.current_frame = None
        self.current_frame_num = None
        self.detection_start_time = None
        self.last_detection_time = None
        self.current_detection_duration = 0.0
        # 👇 КЛЮЧЕВОЙ АТРИБУТ — он будет обновляться в update()
        self.has_pose_landmarks = False

    def update(self, has_pose_landmarks=False, context="", current_frame=None, frame_num=None):
        # 👇 ОБНОВЛЯЕМ АТРИБУТ — ЭТО ВАЖНО!
        self.has_pose_landmarks = has_pose_landmarks

        if has_pose_landmarks:
            if not self.is_detected:
                self.is_detected = True
                self.detection_start_time = time.time()
                self.last_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if self.log_callback:
                    self.log_callback(f"Человек обнаружен ({context})", "SUCCESS")
                if self.autoscreenshot and current_frame is not None:
                    self._save_screenshot(current_frame, "auto_detect")
            else:
                self.current_detection_duration = time.time() - self.detection_start_time
        else:
            if self.is_detected:
                duration = time.time() - self.detection_start_time
                self.detection_history.append({
                    'start_time': self.last_detection_time,
                    'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': duration,
                    'context': context
                })
                self.is_detected = False
                if self.log_callback:
                    self.log_callback(f"Человек покинул кадр ({context}) — длительность: {duration:.1f} сек", "INFO")

        # Сохраняем кадр для отображения (если передан)
        if current_frame is not None:
            self.current_frame = current_frame.copy()

        if frame_num is not None:
            self.current_frame_num = frame_num

    def _save_screenshot(self, frame, prefix):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        success = cv2.imwrite(filepath, frame)
        if self.log_callback and success:
            self.log_callback(f"📸 Автоскриншот: {filepath}", "SUCCESS")
        elif self.log_callback and not success:
            self.log_callback(f"❌ Не удалось сохранить автоскриншот: {filepath}", "ERROR")

    def reset(self):
        """Сбрасывает состояние детектора"""
        self.is_detected = False
        self.detection_start_time = None
        self.current_detection_duration = 0.0
        self.has_pose_landmarks = False