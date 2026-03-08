class BaseDetector:
    name = "base"

    def detect(self, image):
        """
        Returns:
        {
            "people": int,
            "confidence": float
        }
        """
        raise NotImplementedError