import cv2
import os
from person_vs_scenery.base import BaseDetector

# resimde yüz var mı
class HaarDetector(BaseDetector):
    name = "haar"

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cascade_path = os.path.join(base_dir, "models", "haarcascade_frontalface_default.xml")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(60, 60)
        )

        people_count = len(faces)
        confidence = min(1.0, people_count * 0.3)

        return {
            "people": people_count,
            "confidence": confidence
        }
