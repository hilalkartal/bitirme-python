# face_detectors/insightface_face.py
from __future__ import annotations
import numpy as np
from face_detectors.base import BaseFaceDetector

class InsightFaceDetector(BaseFaceDetector):
    name = "insightface"

    def __init__(self, ctx_id: int = -1, det_size=(640, 640), model_name: str = "buffalo_l"):
        """
        ctx_id: -1 CPU, 0 GPU (if onnxruntime-gpu installed + CUDA)
        model_name: common is 'buffalo_l' (det + rec bundle)
        """
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, image):
        # image is BGR (cv2.imread), FaceAnalysis expects BGR as well
        faces = self.app.get(image)

        items = []
        scores = []

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            det_score = float(f.det_score)
            emb = np.asarray(f.embedding, dtype=np.float32)  # (512,)

            items.append({
                "bbox": [x1, y1, x2, y2],
                "det_score": det_score,
                "embedding": emb
            })
            scores.append(det_score)

        avg_conf = float(sum(scores) / len(scores)) if scores else 0.0

        return {
            "faces": items,                 # detailed
            "faces_count": len(items),      # convenience
            "confidence": round(avg_conf, 3)
        }