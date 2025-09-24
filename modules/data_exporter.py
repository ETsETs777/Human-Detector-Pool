# modules/data_exporter.py
import csv
import json
import os
from datetime import datetime

class DataExporter:
    def __init__(self, human_detector, export_dir="exports"):
        self.human_detector = human_detector
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def export_to_csv(self, filename=None):
        if not self.human_detector.detection_history:
            return False, "Нет данных для экспорта."

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.export_dir}/detections_{timestamp}.csv"

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Время начала", "Время окончания", "Длительность (сек)", "Контекст"])
            for entry in self.human_detector.detection_history:
                writer.writerow([
                    entry['start_time'],
                    entry['end_time'],
                    f"{entry['duration']:.2f}",
                    entry['context']
                ])

        return True, f"Экспорт в CSV: {filename}"

    def export_to_json(self, filename=None):
        if not self.human_detector.detection_history:
            return False, "Нет данных для экспорта."

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.export_dir}/detections_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.human_detector.detection_history, f, indent=2, ensure_ascii=False)

        return True, f"Экспорт в JSON: {filename}"

    def export_all(self):
        success_csv, msg_csv = self.export_to_csv()
        success_json, msg_json = self.export_to_json()
        return success_csv and success_json, f"{msg_csv}\n{msg_json}"