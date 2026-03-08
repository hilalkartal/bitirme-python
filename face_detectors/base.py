# face_detectors/base.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

class BaseFaceDetector:
    name = "base-face"

    def detect(self, image) -> Dict[str, Any]:
        """
        Standard return contract (recommended):

        {
            "faces": List[{
                "bbox": [x1, y1, x2, y2],
                "det_score": float,
                "embedding": Optional[np.ndarray]   # only for embedding-capable models
            }],
            "faces_count": int,
            "confidence": float
        }

        Backward compatibility:
        - If a detector only supports counting, it may return:
          { "faces": int, "confidence": float }
        """
        raise NotImplementedError