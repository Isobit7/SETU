"""
SETU — Unified Dynamic 287-Class ISL Recognition Adapter
Consolidates Amrita dynamic conversational phrases (53 classes) and AI4Bharat INCLUDE (263 classes)
into ONE unified dynamic model, deduplicating 29 overlapping signs into 287 unique classes.
Includes negative/resting hand filtering (no_sign) across all 287 signs.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapters.amrita import AmritaPhraseBiLSTM, AmritaPhraseClassifier
from models.adapters.include import (
    INCLUDELstmConfig,
    INCLUDELstmModel,
    INCLUDEInferenceService,
    CandidatePrediction,
    format_display_name,
)

logger = logging.getLogger("setu.models.dynamic_unified")


@dataclass
class UnifiedDynamicResult:
    label: str
    display_name: str
    confidence: float
    top5_candidates: List[CandidatePrediction]
    is_known_sign: bool
    probabilities: Dict[str, float]
    top_3: List[Tuple[str, float]]
    raw_class_idx: int
    source: str
    inference_time_ms: float = 0.0


class UnifiedDynamicAdapter:
    """
    Unified 287-class dynamic ISL sequence recognition adapter.
    Deduplicates the 29 overlapping signs between Amrita and INCLUDE into 287 canonical classes.
    """

    _instance: Optional["UnifiedDynamicAdapter"] = None

    def __init__(
        self,
        label_map_path: Optional[Path] = None,
        amrita_weights_path: Optional[Path] = None,
        include_weights_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent
        self.label_map_path = label_map_path or (
            self.root / "models" / "metadata" / "dynamic_unified_287_label_map.json"
        )
        self.amrita_weights_path = amrita_weights_path or (
            self.root / "models" / "weights" / "isl_dynamic_master_bilstm.pth"
        )
        self.include_weights_path = include_weights_path or (
            self.root / "models" / "pretrained" / "include_no_cnn_lstm.pth"
        )

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.amrita_service: Optional[AmritaPhraseClassifier] = None
        self.include_service: Optional[INCLUDEInferenceService] = None

        self.classes: List[str] = []
        self.display_names: Dict[str, str] = {}
        self.label_to_index: Dict[str, int] = {}
        self.class_details: Dict[str, Any] = {}
        self.num_classes: int = 287

        # Pre-calculated index mappings for fast tensor fusion
        # (amrita_idx -> unified_idx) and (include_idx -> unified_idx)
        self.amrita_to_unified_idx: Dict[int, int] = {}
        self.include_to_unified_idx: Dict[int, int] = {}
        self.duplicate_unified_indices: set = set()
        self.no_sign_idx: int = 0

        self.is_loaded: bool = False
        self._load_metadata()

    @classmethod
    def get_instance(cls) -> "UnifiedDynamicAdapter":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance

    def _load_metadata(self) -> None:
        if not self.label_map_path.exists():
            logger.warning("Unified label map not found at %s", self.label_map_path)
            return

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = data.get("classes", [])
        self.num_classes = len(self.classes)
        self.display_names = data.get("display_names", {})
        self.label_to_index = data.get("label_to_index", {})
        self.class_details = data.get("class_details", {})

        if "no_sign" in self.label_to_index:
            self.no_sign_idx = self.label_to_index["no_sign"]

    def load(self) -> None:
        """Loads both neural submodules and builds fast index lookups."""
        if self.is_loaded:
            return

        # 1. Load Amrita Master 53-class BiLSTM
        try:
            self.amrita_service = AmritaPhraseClassifier(
                weights_path=self.amrita_weights_path,
            )
            if hasattr(self.amrita_service.model, "to"):
                self.amrita_service.model.to(self.device)
            logger.info("Loaded Amrita dynamic backbone on %s", self.device)
        except Exception as e:
            logger.error("Failed to load Amrita backbone: %s", e)

        # 2. Load INCLUDE 263-class BiLSTM
        try:
            self.include_service = INCLUDEInferenceService.get_instance()
            self.include_service.weights_path = self.include_weights_path
            self.include_service.device = self.device
            self.include_service.load()
            logger.info("Loaded INCLUDE dynamic backbone on %s", self.device)
        except Exception as e:
            logger.error("Failed to load INCLUDE backbone: %s", e)

        # 3. Build fast cross-index maps
        if self.amrita_service and self.amrita_service.class_names:
            for a_idx, a_name in enumerate(self.amrita_service.class_names):
                norm = a_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                for u_name, u_idx in self.label_to_index.items():
                    u_norm = u_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                    if norm == u_norm:
                        self.amrita_to_unified_idx[a_idx] = u_idx
                        break

        if self.include_service and self.include_service.idx_to_label:
            for inc_idx, inc_name in self.include_service.idx_to_label.items():
                norm = inc_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                for u_name, u_idx in self.label_to_index.items():
                    u_norm = u_name.lower().replace(" ", "").replace("_", "").replace("-", "")
                    if norm == u_norm:
                        self.include_to_unified_idx[inc_idx] = u_idx
                        if u_idx in self.amrita_to_unified_idx.values():
                            self.duplicate_unified_indices.add(u_idx)
                        break

        self.is_loaded = True
        logger.info(
            "UnifiedDynamicAdapter initialized: %d classes (%d duplicates deduplicated, mapped %d Amrita, %d INCLUDE)",
            self.num_classes,
            len(self.duplicate_unified_indices),
            len(self.amrita_to_unified_idx),
            len(self.include_to_unified_idx),
        )

    def predict(
        self,
        features_amrita_316: Optional[np.ndarray] = None,
        features_include_134: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.35,
    ) -> UnifiedDynamicResult:
        """
        Runs unified 287-class dynamic recognition.
        - features_amrita_316: (T, 316) array of coordinates + velocities
        - features_include_134: (T, 134) array of pose + dual-hand landmarks
        Fuses predictions, eliminates duplicates, and applies negative resting hand gating.
        """
        if not self.is_loaded:
            self.load()

        # Handle case where features_amrita_316 contains 134-D INCLUDE features
        if features_amrita_316 is not None and features_include_134 is None:
            if hasattr(features_amrita_316, "shape") and len(features_amrita_316.shape) == 2 and features_amrita_316.shape[1] == 134:
                features_include_134 = features_amrita_316
                features_amrita_316 = None

        # Guard: If no landmarks or all-zero features, immediately return resting idle prediction
        has_amrita = features_amrita_316 is not None and not np.all(np.abs(features_amrita_316) < 1e-4)
        has_include = features_include_134 is not None and not np.all(np.abs(features_include_134) < 1e-4)
        if not has_amrita and not has_include:
            return self._empty_prediction()

        t0 = time.perf_counter()
        unified_probs = np.zeros(self.num_classes, dtype=np.float32)

        p_no_sign = 0.0
        amrita_probs: Optional[np.ndarray] = None
        include_probs: Optional[np.ndarray] = None

        # 1. Forward Amrita Dynamic Model
        if self.amrita_service and features_amrita_316 is not None and len(features_amrita_316) >= 5:
            try:
                arr = features_amrita_316
                if arr.shape[1] < 316:
                    pad = 316 - arr.shape[1]
                    arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
                else:
                    arr = arr[:, :316]
                res = self.amrita_service.predict(arr, confidence_threshold=0.0)
                p_no_sign = float(res.probabilities.get("no_sign", 0.0))
                amrita_probs = np.array(
                    [res.probabilities.get(c, 0.0) for c in self.amrita_service.class_names],
                    dtype=np.float32,
                )
            except Exception as e:
                logger.debug("Amrita forward error in unified model: %s", e)

        # 2. Forward INCLUDE Model
        if self.include_service and features_include_134 is not None and len(features_include_134) >= 12:
            try:
                # Expects (T, 134)
                arr = features_include_134
                if arr.ndim == 2 and arr.shape[1] == 134:
                    seq_len = arr.shape[0]
                    if seq_len < 200:
                        arr = np.pad(arr, ((0, 200 - seq_len), (0, 0)), "constant")
                    elif seq_len > 200:
                        arr = arr[:200]
                    tensor_in = torch.from_numpy(arr).unsqueeze(0).float().to(self.device)
                    with torch.no_grad():
                        logits = self.include_service.model(tensor_in)
                        include_probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            except Exception as e:
                logger.debug("INCLUDE forward error in unified model: %s", e)

        # 3. Probability Fusion and Deduplication
        # If resting hand detected by Amrita no_sign, suppress active sign probabilities
        suppression_factor = max(0.0, 1.0 - (p_no_sign * 1.5))

        # Map Amrita probabilities
        if amrita_probs is not None:
            for a_idx, a_prob in enumerate(amrita_probs):
                u_idx = self.amrita_to_unified_idx.get(a_idx)
                if u_idx is not None:
                    if u_idx == self.no_sign_idx:
                        unified_probs[u_idx] = float(a_prob)
                    else:
                        unified_probs[u_idx] = float(a_prob * suppression_factor)

        # Map INCLUDE probabilities
        if include_probs is not None:
            for inc_idx, inc_prob in enumerate(include_probs):
                u_idx = self.include_to_unified_idx.get(inc_idx)
                if u_idx is not None:
                    scaled_inc = float(inc_prob * suppression_factor)
                    if u_idx in self.duplicate_unified_indices:
                        # Deduplicated overlapping sign: fuse by max confidence
                        unified_probs[u_idx] = max(unified_probs[u_idx], scaled_inc)
                    else:
                        unified_probs[u_idx] = scaled_inc

        # 4. Extract Top Candidates
        top5_indices = np.argsort(unified_probs)[::-1][:5]
        top1_idx = int(top5_indices[0])
        top1_label = self.classes[top1_idx] if self.classes else "unknown"
        top1_conf = float(unified_probs[top1_idx])

        # Source attribution
        if top1_idx in self.duplicate_unified_indices:
            pred_source = "fused (Amrita + INCLUDE)"
        elif top1_idx in self.amrita_to_unified_idx.values():
            pred_source = "amrita_dynamic"
        else:
            pred_source = "include_263"

        top5 = [
            CandidatePrediction(
                label=self.classes[int(idx)],
                display_name=self.display_names.get(self.classes[int(idx)], format_display_name(self.classes[int(idx)])),
                confidence=float(unified_probs[int(idx)]),
            )
            for idx in top5_indices
        ]

        top_3 = [
            (c.display_name, round(c.confidence, 4))
            for c in top5[:3]
        ]

        probs_dict = {
            self.classes[int(idx)]: round(float(unified_probs[int(idx)]), 4)
            for idx in top5_indices
        }

        # Sign validity
        is_known = (
            top1_conf >= confidence_threshold
            and top1_label != "no_sign"
            and p_no_sign < 0.50
        )

        display_name = self.display_names.get(top1_label, format_display_name(top1_label))
        if not is_known:
            display_name = "Waiting for a sign..."

        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        return UnifiedDynamicResult(
            label=top1_label if is_known else "no_sign",
            display_name=display_name,
            confidence=top1_conf,
            top5_candidates=top5,
            is_known_sign=is_known,
            probabilities=probs_dict,
            top_3=top_3,
            raw_class_idx=top1_idx,
            source=pred_source,
            inference_time_ms=inference_time_ms,
        )

    def _empty_prediction(self) -> UnifiedDynamicResult:
        return UnifiedDynamicResult(
            label="no_sign",
            display_name="Waiting for a sign...",
            confidence=0.0,
            top5_candidates=[],
            is_known_sign=False,
            probabilities={},
            top_3=[],
            raw_class_idx=self.no_sign_idx,
            source="idle",
            inference_time_ms=0.0,
        )
