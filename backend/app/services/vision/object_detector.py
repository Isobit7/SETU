"""
SETU — High-Precision Multimodal Object Detector Service
Detects physical objects presented in front of the camera (e.g. wristwatches,
smartwatches, cell phones, water bottles, cups, books, pens, scissors) for
Show-and-Sign multimodal ISL interaction.

Powered by Open-Vocabulary YOLO-World (yolov8s-worldv2, ~13M params, 24MB)
with zero-shot prompt grounding, direct wristwatch vs phone separation,
and graceful fallback to torchvision FasterRCNN-FPN / SSDLite.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("setu.services.object_detector")

# Custom open-vocabulary classes for Show-and-Sign interaction
YOLO_WORLD_CLASSES: List[str] = [
    "wristwatch",
    "smartwatch",
    "cell phone",
    "water bottle",
    "coffee cup",
    "mug",
    "book",
    "pen",
    "scissors",
    "keys",
    "wallet",
    "laptop",
    "apple",
    "banana",
    "orange",
    "glasses",
]

# Friendly mappings for detected classes
LABEL_NORMALIZATION: Dict[str, str] = {
    "wristwatch": "watch",
    "smartwatch": "watch",
    "cell phone": "phone",
    "water bottle": "bottle",
    "coffee cup": "cup",
    "mug": "cup",
    "wine glass": "glass",
    "dining table": "table",
    "potted plant": "plant",
    "hair drier": "hairdryer",
}

# Salient interactive objects relevant to Show-and-Sign
INTERACTIVE_OBJECT_CLASSES = {
    "watch", "wristwatch", "smartwatch",
    "bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "book", "cell phone", "phone", "laptop", "mouse", "remote", "keyboard",
    "scissors", "toothbrush", "pen", "backpack", "umbrella", "handbag",
    "clock", "vase", "teddy bear", "sports ball", "frisbee", "box",
    "keys", "wallet", "glasses",
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
    Uses open-vocabulary YOLO-World (yolov8s-worldv2) when available for accurate
    everyday object detection (including watches, phones, bottles, cups, books, pens)
    without MS-COCO vocabulary limitations (which misclassified watches as phones).
    Falls back gracefully to FasterRCNN-FPN / SSDLite.
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

        self._yolo_model = None
        self._fallback_model = None
        self._model_name = "None"
        self.categories: List[str] = []

        # 1. Attempt loading lightweight Open-Vocabulary YOLO-World (13M params, 24MB)
        # Directly resolves MS-COCO vocabulary void (accurately separates wristwatch vs cell phone)
        try:
            from ultralytics import YOLOWorld
            model_candidates = [
                Path("models/weights/yolov8s-worldv2.pt"),
                Path(__file__).resolve().parents[4] / "models" / "weights" / "yolov8s-worldv2.pt",
                Path("yolov8s-worldv2.pt"),
                Path(__file__).resolve().parents[4] / "yolov8s-worldv2.pt",
            ]
            yolo_path = next((p for p in model_candidates if p.is_file()), None)
            if yolo_path is not None or Path("yolov8s-worldv2.pt").exists():
                chosen_path = str(yolo_path) if yolo_path else "yolov8s-worldv2.pt"
                yolo = YOLOWorld(chosen_path)
                yolo.set_classes(YOLO_WORLD_CLASSES)
                self._yolo_model = yolo
                self._model_name = "YOLO-World-v2-Small"
                self.categories = list(YOLO_WORLD_CLASSES)
                logger.info("Loaded open-vocabulary YOLO-World (%d classes) from %s", len(self.categories), chosen_path)
        except Exception as e:
            logger.warning("YOLO-World not loaded, falling back to torchvision: %s", e)

        # 2. Fallback to torchvision FasterRCNN / SSDLite if YOLO-World is unavailable
        if self._yolo_model is None:
            from torchvision.models.detection import (
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                SSDLite320_MobileNet_V3_Large_Weights,
                fasterrcnn_mobilenet_v3_large_320_fpn,
                ssdlite320_mobilenet_v3_large,
            )
            try:
                self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                self.categories = self.weights.meta.get("categories", [])
                self._fallback_model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
                self._model_name = "FasterRCNN-FPN-320"
            except Exception as e:
                logger.warning("Falling back to SSDLite detector: %s", e)
                self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                self.categories = self.weights.meta.get("categories", [])
                self._fallback_model = ssdlite320_mobilenet_v3_large(weights=self.weights)
                self._model_name = "SSDLite320"

            self._fallback_model.to(self.device)
            self._fallback_model.eval()

        logger.info(
            "COCOObjectDetector [%s] initialized on %s (%d categories). Base confidence threshold: %.2f",
            self._model_name,
            self.device,
            len(self.categories),
            self.confidence_threshold,
        )

    def warmup(self) -> None:
        """Primes compute graphs."""
        if self._yolo_model is not None:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            try:
                dev = str(self.device) if self.device.type == "cuda" else "cpu"
                _ = self._yolo_model.predict(dummy, conf=0.25, verbose=False, device=dev)
            except Exception as e:
                logger.debug("YOLO warmup warning: %s", e)
        elif self._fallback_model is not None:
            dummy = torch.rand(1, 3, 320, 320, device=self.device)
            with torch.inference_mode():
                for _ in range(2):
                    _ = self._fallback_model(dummy)

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

        boxes = np.empty((0, 4))
        scores = np.empty((0,))
        labels = np.empty((0,), dtype=int)
        cat_lookup: Dict[int, str] = {}

        # 1. Primary Engine: YOLO-World
        if self._yolo_model is not None:
            det_conf = min(thresh, 0.35) if hand_landmarks else min(thresh, 0.45)
            dev = str(self.device) if self.device.type == "cuda" else "cpu"
            preds = self._yolo_model.predict(frame, conf=det_conf, verbose=False, device=dev)[0]
            if len(preds.boxes) > 0:
                boxes = preds.boxes.xyxy.cpu().numpy()
                scores = preds.boxes.conf.cpu().numpy()
                labels = preds.boxes.cls.cpu().numpy().astype(int)
            cat_lookup = self._yolo_model.names

        # 2. Fallback Engine: FasterRCNN / SSDLite
        elif self._fallback_model is not None:
            if frame.ndim == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame
            img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
            predictions = self._fallback_model(img_tensor)[0]
            boxes = predictions["boxes"].cpu().numpy()
            scores = predictions["scores"].cpu().numpy()
            labels = predictions["labels"].cpu().numpy()
            cat_lookup = {i: c for i, c in enumerate(self.categories)}

        results: List[DetectedObject] = []

        for box, score, lbl_idx in zip(boxes, scores, labels):
            score_f = float(score)

            cat_name = cat_lookup.get(int(lbl_idx), "unknown") if isinstance(cat_lookup, dict) else "unknown"

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

            # Hand proximity check
            is_held = False
            if hand_landmarks:
                for pt in hand_landmarks:
                    px = pt.get("x", 0.0) if isinstance(pt, dict) else getattr(pt, "x", 0.0)
                    py = pt.get("y", 0.0) if isinstance(pt, dict) else getattr(pt, "y", 0.0)
                    # Point in or near bounding box with 6% margin
                    if (norm_box[0] - 0.06 <= py <= norm_box[2] + 0.06) and (
                        norm_box[1] - 0.06 <= px <= norm_box[3] + 0.06
                    ):
                        is_held = True
                        break

            # 1. Reject tiny speck noise:
            # When held in hands, allow smaller handheld objects down to 1.0% area (watches, pens, keys)
            # When not held, reject boxes < 3.0% area
            min_area = 0.010 if is_held else 0.030
            if box_area < min_area or max(box_w, box_h) < 0.05:
                continue

            # 2. Strict Watch vs Phone disambiguation on fallback:
            # In old COCO models without "watch", wrist false positives had tiny areas
            if self._yolo_model is None and norm_name == "phone" and box_area < 0.05:
                continue

            # 3. If not held in hands, require higher certainty to prevent passive background clutter
            if not is_held and score_f < 0.55:
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
