"""
SETU — Instant Few-Shot Custom Sign Personalization Service
============================================================
Enables instantaneous personalization for signers whose unique signing morphology,
lighting, or camera angle challenges generalized neural network models.

Loads user-recorded exemplar sequences from `data/processed/custom_*.npz`,
builds normalized spatial-temporal prototypes, and performs sub-millisecond
cosine similarity matching during live WebSocket recognition.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("setu.services.few_shot")


class FewShotCustomSignService:
    """
    Sub-millisecond personal few-shot nearest-prototype matcher.
    
    When general models have borderline confidence on a user's sign,
    this service matches against the user's recorded personal exemplars.
    """

    def __init__(self, data_dir: Path = Path("data/processed")) -> None:
        self.data_dir = data_dir
        self.prototypes: Dict[str, np.ndarray] = {}  # label -> normalized 164-D or 126-D vector
        self.raw_prototypes_30x164: Dict[str, List[np.ndarray]] = {}
        self.last_reload_time: float = 0.0
        self.reload()

    def reload(self) -> int:
        """Scans data/processed/ for custom_*.npz files and builds exemplar prototypes."""
        self.prototypes.clear()
        self.raw_prototypes_30x164.clear()
        
        if not self.data_dir.exists():
            return 0

        sample_files = list(self.data_dir.glob("custom_*.npz"))
        if not sample_files:
            return 0

        grouped_features: Dict[str, List[np.ndarray]] = {}

        for sf in sample_files:
            try:
                data = np.load(sf)
                label = str(data.get("label", "")).strip().lower()
                feats = data.get("features")
                if not label or feats is None:
                    continue
                # feats is (30, 164)
                if feats.ndim == 2 and feats.shape[0] == 30:
                    grouped_features.setdefault(label, []).append(feats.astype(np.float32))
            except Exception as e:
                logger.debug("Failed reading custom sample %s: %s", sf.name, e)

        for label, seq_list in grouped_features.items():
            self.raw_prototypes_30x164[label] = seq_list
            # Compute mean representative vector across all 30 frames
            # Middle stroke phase (frames 8 to 22) represents the stable apex of the sign
            mid_frames = [seq[8:23].mean(axis=0) for seq in seq_list]
            mean_proto = np.mean(mid_frames, axis=0)  # Shape (164,)
            norm = np.linalg.norm(mean_proto)
            if norm > 1e-4:
                self.prototypes[label] = mean_proto / norm

        self.last_reload_time = time.time()
        logger.info(
            "FewShotCustomSignService loaded %d custom sign prototypes (%s).",
            len(self.prototypes),
            ", ".join(self.prototypes.keys()) if self.prototypes else "None",
        )
        return len(self.prototypes)

    def match(
        self,
        current_feat_164: np.ndarray,
        threshold: float = 0.88,
    ) -> Optional[Tuple[str, float, str]]:
        """
        Matches an incoming 164-D feature vector against stored personal prototypes.
        
        Args:
            current_feat_164: (164,) feature vector from current frame or rolling window.
            threshold: Minimum cosine similarity required to trigger a custom match.
            
        Returns:
            (matched_label, similarity_score, display_name) if matched, else None.
        """
        if not self.prototypes or current_feat_164 is None:
            return None

        # Check dimension
        if current_feat_164.ndim != 1 or current_feat_164.shape[0] < 158:
            return None

        feat_vector = current_feat_164[:164] if current_feat_164.shape[0] >= 164 else current_feat_164
        norm = np.linalg.norm(feat_vector)
        if norm < 1e-4:
            return None

        unit_vec = feat_vector / norm
        best_label: Optional[str] = None
        best_score: float = -1.0

        for label, proto in self.prototypes.items():
            # Align dimension if prototype is 164 and input is 158 or vice versa
            min_len = min(len(unit_vec), len(proto))
            score = float(np.dot(unit_vec[:min_len], proto[:min_len]))
            if score > best_score:
                best_score = score
                best_label = label

        if best_label and best_score >= threshold:
            display_name = best_label.upper().replace("_", " ")
            return best_label, best_score, display_name

        return None


# Global singleton instance
custom_sign_service = FewShotCustomSignService()
