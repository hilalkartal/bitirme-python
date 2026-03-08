from ultralytics import YOLO
from person_vs_scenery.base import BaseDetector
import os


# resimde insan var mı
class YOLOv8Detector(BaseDetector):
    name = "yolov8"

    def __init__(self):
        # pretrained YOLOv8 model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "yolov8n.pt")

        self.model = YOLO(model_path)

    def detect(self, image):
        # Run inference
        results = self.model(image, verbose=False)

        people_count = 0
        confidences = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 0 = person
                if cls_id == 0:
                    people_count += 1
                    confidences.append(conf)

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        return {
            "people": people_count,
            "confidence": round(avg_confidence, 3)
        }
