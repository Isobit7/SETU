"""
SETU — Multimodal 80+ COCO Object Detector Service
Detects physical objects presented in front of the camera (e.g. water bottles,
cups, cell phones, books, scissors, fruit) for Show-and-Sign multimodal ISL interaction.
Powered by SSDLite320-MobileNetV3 on PyTorch/CUDA with sub-25ms latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

logger = logging.getLogger("setu.services.object_detector")

# Friendly mappings for common COCO categories
LABEL_NORMALIZATION: Dict[str, str] = {
    "cell phone": "phone",
    "wine glass": "glass",
    "dining table": "table",
    "potted plant": "plant",
    "hair drier": "hairdryer",
}

# Salient interactive objects relevant to Show-and-Sign
INTERACTIVE_OBJECT_CLASSES = {
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "book", "cell phone", "laptop", "mouse", "remote", "keyboard",
    "scissors", "toothbrush", "pen", "backpack", "umbrella", "handbag",
    "clock", "vase", "teddy bear", "sports ball", "frisbee", "box",
}


@dataclass(frozen=True)
class DetectedObject:
    """Represents a detected object in the camera frame."""
    label: str
    display_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] normalized [0, 1]
    is_interactive: bool = False
    is_held: bool = False


class COCOObjectDetector:
    """
    Lightweight, fast object detector covering standard MS-COCO classes.
    Runs asynchronously or periodically to detect objects held by the user.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.categories: List[str] = self.weights.meta.get("categories", [])
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "COCOObjectDetector initialized on %s (%d categories). Confidence threshold: %.2f",
            self.device,
            len(self.categories),
            self.confidence_threshold,
        )

    def warmup(self) -> None:
        """Primes PyTorch/CUDA compute graphs."""
        dummy = torch.rand(1, 3, 320, 320, device=self.device)
        with torch.inference_mode():
            for _ in range(2):
                _ = self.model(dummy)

    @torch.inference_mode()
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        hand_landmarks: Optional[List[Dict[str, float]]] = None,
    ) -> List[DetectedObject]:
        """
        Detects objects in an RGB/BGR OpenCV frame.
        
        Args:
            frame: (H, W, 3) BGR or RGB image.
            confidence_threshold: Override confidence threshold.
            hand_landmarks: Optional list of hand landmark dicts with 'x', 'y' for hand-proximity check.
            
        Returns:
            List of DetectedObject records sorted by confidence.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Convert to RGB float tensor [0, 1] normalized (C, H, W)
        if frame.ndim == 3 and frame.shape[2] == 3:
            # OpenCV frames are typically BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        predictions = self.model(img_tensor)[0]
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        results: List[DetectedObject] = []

        for box, score, lbl_idx in zip(boxes, scores, labels):
            score_f = float(score)
            if score_f < thresh:
                continue

            # In torchvision detection models, category index is 0-indexed or 1-indexed
            cat_name = "unknown"
            if 0 <= lbl_idx < len(self.categories):
                cat_name = self.categories[lbl_idx]

            # Filter out person detection (the user) from the object catalog
            if cat_name.lower() in ("person", "n/a", "unknown", ""):
                continue

            norm_name = LABEL_NORMALIZATION.get(cat_name.lower(), cat_name.lower())
            disp_name = norm_name.upper().replace("_", " ")

            # Bounding box in normalized [ymin, xmin, ymax, xmax]
            xmin, ymin, xmax, ymax = box
            norm_box = (
                float(np.clip(ymin / h, 0.0, 1.0)),
                float(np.clip(xmin / w, 0.0, 1.0)),
                float(np.clip(ymax / h, 0.0, 1.0)),
                float(np.clip(xmax / w, 0.0, 1.0)),
            )

            # Check if this object is in the user's hand interaction zone
            is_held = False
            if hand_landmarks:
                for pt in hand_landmarks:
                    px = pt.get("x", 0.0)
                    py = pt.get("y", 0.0)
                    # Point in or near bounding box with 5% margin
                    if (norm_box[0] - 0.08 <= py <= norm_box[2] + 0.08) and (
                        norm_box[1] - 0.08 <= px <= norm_box[3] + 0.08
                    ):
                        is_held = True
                        break

            is_interactive = norm_name in INTERACTIVE_OBJECT_CLASSES

            results.append(
                DetectedObject(
                    label=norm_name,
                    display_name=disp_name,
                    confidence=round(score_f, 4),
                    bbox=norm_box,
                    is_interactive=is_interactive,
                    is_held=is_held,
                )
            )

        # Sort: held objects first, then by confidence descending
        results.sort(key=lambda o: (1 if o.is_held else 0, o.confidence), reverse=True)
        return results

