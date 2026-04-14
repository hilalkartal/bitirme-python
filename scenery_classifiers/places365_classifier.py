"""
ResNet18 scene classifier pretrained on Places365.

MIT's official Places365 model hub only publishes ResNet/AlexNet/VGG/DenseNet
weights — MobileNetV2 is not officially available.  ResNet18 is the lightest
of the officially supported models:

  File on disk : ~45 MB
  RAM at rest  : ~95 MB   (model tensors)
  Peak inference: ~120 MB  (activations, freed right after forward pass)

Compare: InsightFace (buffalo_l) ≈ 300-400 MB.

Loading is LAZY — the model is not instantiated until the first scenery
image is processed, so people-only sessions pay zero extra RAM cost.

The 365 Places categories cover a wide range of indoor/outdoor scenes:
beach, mountain, forest, street, bedroom, kitchen, office, etc.
"""

from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote asset URLs  (official MIT CSAIL Places365 mirror)
# ---------------------------------------------------------------------------
_WEIGHTS_URL = (
    "http://places2.csail.mit.edu/models_places365/"
    "resnet18_places365.pth.tar"
)
_LABELS_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/"
    "categories_places365.txt"
)

# ---------------------------------------------------------------------------
# Paths: project_root/models/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "models")
_WEIGHTS_PATH = os.path.join(_MODELS_DIR, "resnet18_places365.pth.tar")
_LABELS_PATH  = os.path.join(_MODELS_DIR, "categories_places365.txt")


def _download_if_missing(url: str, dest: str, description: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logger.info("Downloading %s (~45 MB) from %s …", description, url)
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        logger.info("Saved %s → %s", description, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _load_labels(path: str) -> list[str]:
    """
    Parse categories_places365.txt.
    Each line:  /a/abbey 0
    Returns plain names like 'abbey', 'beach', 'mountain snowy'.
    """
    labels: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = line.rsplit(" ", 1)[0]       # '/a/abbey'
            name = raw.split("/")[-1]           # 'abbey'
            labels.append(name.replace("_", " "))
    return labels


class Places365Classifier:
    """
    Lazy-loading ResNet18 scene classifier (Places365 weights).

    Usage
    -----
    classifier = Places365Classifier()             # nothing loaded yet
    tags = classifier.classify(bgr_image, top_k=3)
    # tags -> [{"label": "beach", "confidence": 0.38}, ...]
    """

    name = "resnet18-places365"

    def __init__(self) -> None:
        self._model      = None
        self._labels: list[str] | None = None
        self._transform  = None

    # ------------------------------------------------------------------
    # Internal: lazy initialiser
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Heavy imports only when we actually need the model
        import torch
        import torchvision.models as tv_models
        import torchvision.transforms as T

        # 1. Download assets on first use
        _download_if_missing(_WEIGHTS_URL, _WEIGHTS_PATH, "ResNet18 Places365 weights")
        _download_if_missing(_LABELS_URL,  _LABELS_PATH,  "Places365 category labels")

        # 2. Parse labels
        self._labels = _load_labels(_LABELS_PATH)
        if len(self._labels) != 365:
            logger.warning(
                "Expected 365 Places365 labels, got %d — label file may be malformed",
                len(self._labels),
            )

        # 3. Build ResNet18 with a 365-class output head
        model = tv_models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 365)

        # 4. Load Places365 checkpoint
        checkpoint = torch.load(_WEIGHTS_PATH, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip 'module.' prefix added by DataParallel training
        clean: dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean)
        model.eval()

        # 5. Preprocessing (standard ImageNet normalisation, 224x224)
        self._transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self._model = model
        logger.info("ResNet18 Places365 loaded — ready for scene classification")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(
        self,
        image: np.ndarray,
        top_k: int = 3,
        min_confidence: float = 0.10,
    ) -> list[dict]:
        """
        Classify the scene in an OpenCV BGR image.

        Parameters
        ----------
        image          : np.ndarray  OpenCV BGR image (HxWx3 uint8)
        top_k          : int         Maximum number of predictions to return
        min_confidence : float       Discard predictions below this probability
                                     (Places365 has 365 classes so raw softmax
                                      scores are naturally lower than ImageNet)

        Returns
        -------
        List of dicts sorted by confidence descending:
            [{"label": "beach", "confidence": 0.38}, ...]
        Empty list if no prediction exceeds min_confidence.
        """
        self._ensure_loaded()

        import torch
        from PIL import Image as PILImage

        # OpenCV BGR -> RGB -> PIL
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        tensor = self._transform(pil_img).unsqueeze(0)   # (1, 3, 224, 224)

        with torch.no_grad():
            logits = self._model(tensor)                 # (1, 365)
            probs  = torch.softmax(logits, dim=1)[0]     # (365,)

        top_probs, top_indices = probs.topk(top_k)

        results: list[dict] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            if prob < min_confidence:
                continue
            label = self._labels[idx]
            results.append({"label": label, "confidence": round(prob, 3)})

        logger.info(
            "Scene classification: %s",
            [(r["label"], r["confidence"]) for r in results],
        )
        return results
