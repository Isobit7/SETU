"""
SETU — High-Precision Multimodal COCO Object Detector Service
Detects physical objects presented in front of the camera (e.g. water bottles,
cups, cell phones, books, scissors, fruit) for Show-and-Sign multimodal ISL interaction.
Powered by FasterRCNN-FPN / SSDLite on PyTorch/CUDA with strict confidence gating,
bounding-box scale filtering, and wrist-watch hallucination suppression.
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
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
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
    High-precision object detector for Show-and-Sign interaction.
    Filters out background clutter, watches worn on wrists, and false alarms.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try FasterRCNN with FPN feature pyramids first for superior precision; fallback to SSDLite
        try:
            self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.categories: List[str] = self.weights.meta.get("categories", [])
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
            self._model_name = "FasterRCNN-FPN-320"
        except Exception as e:
            logger.warning("Falling back to SSDLite detector: %s", e)
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.categories = self.weights.meta.get("categories", [])
            self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
            self._model_name = "SSDLite320"

        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "COCOObjectDetector [%s] initialized on %s (%d categories). Confidence threshold: %.2f",
            self._model_name,
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
        Detects salient objects in an RGB/BGR frame with strict filtering against false positives.
        
        Args:
            frame: (H, W, 3) image.
            confidence_threshold: Override confidence threshold (defaults to 0.65).
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

            cat_name = "unknown"
            if 0 <= lbl_idx < len(self.categories):
                cat_name = self.categories[lbl_idx]

            # Filter out person detection (the user) from the object catalog
            if cat_name.lower() in ("person", "n/a", "unknown", ""):
                continue

            norm_name = LABEL_NORMALIZATION.get(cat_name.lower(), cat_name.lower())
            disp_name = norm_name.upper().replace("_", " ")

            # Only accept classes designated for Show-and-Sign interaction
            if norm_name not in INTERACTIVE_OBJECT_CLASSES and cat_name.lower() not in INTERACTIVE_OBJECT_CLASSES:
                continue

            xmin, ymin, xmax, ymax = box
            norm_box = (
                float(np.clip(ymin / h, 0.0, 1.0)),
                float(np.clip(xmin / w, 0.0, 1.0)),
                float(np.clip(ymax / h, 0.0, 1.0)),
                float(np.clip(xmax / w, 0.0, 1.0)),
            )

            box_h = norm_box[2] - norm_box[0]
            box_w = norm_box[3] - norm_box[1]
            box_area = box_h * box_w

            # 1. Reject tiny bounding boxes (< 3.5% area or < 8% dimension)
            # This completely suppresses wristwatches, bracelets, buttons, rings, and background specks
            if box_area < 0.035 or box_w < 0.08 or box_h < 0.08:
                continue

            # 2. Strict Watch vs Phone suppression:
            # When people wear a watch, false "cell phone" boxes on the wrist have area < 0.05
            if norm_name in ("phone", "cell phone") and box_area < 0.05:
                continue

            # 3. Check proximity to user's hands
            is_held = False
            if hand_landmarks:
                for pt in hand_landmarks:
                    px = pt.get("x", 0.0)
                    py = pt.get("y", 0.0)
                    # Point in or near bounding box with 6% margin
                    if (norm_box[0] - 0.06 <= py <= norm_box[2] + 0.06) and (
                        norm_box[1] - 0.06 <= px <= norm_box[3] + 0.06
                    ):
                        is_held = True
                        break

            # If not held in hands, require high certainty (>= 0.78) to prevent passive background objects
            if not is_held and score_f < 0.78:
                continue

            results.append(
                DetectedObject(
                    label=norm_name,
                    display_name=disp_name,
                    confidence=round(score_f, 4),
                    bbox=norm_box,
                    is_interactive=True,
                    is_held=is_held,
                )
            )

        # Sort: held objects first, then by confidence descending
        results.sort(key=lambda o: (1 if o.is_held else 0, o.confidence), reverse=True)
        return results
