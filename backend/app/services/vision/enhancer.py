"""
SETU — Adaptive Low-Light & Contrast Video Enhancer
Improves landmark tracking robustness in dim environments or poor webcam conditions
using Contrast Limited Adaptive Histogram Equalization (CLAHE) and bilateral noise reduction.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("setu.services.enhancer")


class AdaptiveLowLightEnhancer:
    """
    Real-time adaptive image enhancer for dim lighting conditions.
    Operates on LAB color space to boost luminance and local contrast while
    preserving color fidelity and suppressing high-frequency sensor noise.
    """

    def __init__(
        self,
        low_light_threshold: float = 75.0,
        clip_limit: float = 2.5,
        grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.low_light_threshold = low_light_threshold
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Evaluates frame brightness and applies adaptive enhancement if dim.
        
        Args:
            frame: (H, W, 3) BGR uint8 OpenCV image.
            
        Returns:
            Tuple of:
            - processed_frame: np.ndarray (enhanced if low light, otherwise original)
            - lighting_score: float normalized [0.0 (pitch black) to 1.0 (optimal/bright)]
            - was_enhanced: bool indicating if CLAHE enhancement was applied
        """
        if frame is None or frame.size == 0:
            return frame, 0.0, False

        # Convert to LAB color space to extract luminance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        mean_l = float(np.mean(l_channel))
        # Lighting score: 0 = completely dark, 128+ = well lit (normalized to 1.0)
        lighting_score = round(min(mean_l / 128.0, 1.0), 3)

        if mean_l < self.low_light_threshold:
            # Apply CLAHE to L channel to restore contrast and visibility
            enhanced_l = self.clahe.apply(l_channel)

            # Subtle bilateral filter on enhanced luminance to smooth camera thermal grain
            enhanced_l = cv2.bilateralFilter(enhanced_l, d=5, sigmaColor=35, sigmaSpace=35)

            merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
            return enhanced_bgr, lighting_score, True

        return frame, lighting_score, False

