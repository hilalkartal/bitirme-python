import mediapipe as mp
import cv2
from face_detectors.base import BaseFaceDetector

class MediaPipeFaceDetector(BaseFaceDetector):
    name = "mediapipe-face"

    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        items = []
        scores = []

        if results.detections:
            h, w = image.shape[:2]
            for det in results.detections:
                score = float(det.score[0])
                bbox = det.location_data.relative_bounding_box

                # convert relative bbox -> absolute (x1,y1,x2,y2)
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                items.append({
                    "bbox": [x1, y1, x2, y2],
                    "det_score": score,
                    "embedding": None
                })
                scores.append(score)

        avg_conf = float(sum(scores) / len(scores)) if scores else 0.0

        return {
            "faces": items,
            "faces_count": len(items),
            "confidence": round(avg_conf, 3)
        }