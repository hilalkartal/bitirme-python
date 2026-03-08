import cv2
from person_vs_scenery.base import BaseDetector

#resimde insan var mı
class HOGDetector(BaseDetector):
    name = "hog"

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        boxes, _ = self.hog.detectMultiScale(image)

        people_count = len(boxes)
        confidence = min(1.0, people_count * 0.4)

        return {
            "people": people_count,
            "confidence": confidence
        }
