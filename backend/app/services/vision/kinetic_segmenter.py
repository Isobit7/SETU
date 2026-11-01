"""
SETU — Kinetic Energy Wavelet Gesture Segmenter.

Monitors spatial joint velocities to detect authentic sign gesture boundaries
(onset, active trajectory, deceleration, and settled completion) in real-time video streams.
Prevents spurious mid-movement classifications and handles entry/exit transients.
"""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class GestureState(str, Enum):
    IDLE = "IDLE"
    GESTURE_START = "GESTURE_START"
    GESTURE_ACTIVE = "GESTURE_ACTIVE"
    GESTURE_SETTLED = "GESTURE_SETTLED"


class KineticEnergySegmenter:
    """
    State machine that monitors hand motion energy to segment continuous video
    into discrete gesture sequences with adaptive baseline noise calibration.
    """

    def __init__(
        self,
        energy_threshold: float = 0.025,
        settle_window_frames: int = 3,
        max_gesture_frames: int = 60,
        min_gesture_frames: int = 6,
        alpha_smooth: float = 0.35,
    ) -> None:
        self.energy_threshold: float = float(energy_threshold)
        self.settle_window_frames: int = int(settle_window_frames)
        self.max_gesture_frames: int = int(max_gesture_frames)
        self.min_gesture_frames: int = int(min_gesture_frames)
        self.alpha_smooth: float = float(alpha_smooth)

        self.state: GestureState = GestureState.IDLE
        self.smoothed_energy: float = 0.0
        self.prev_pts: Optional[np.ndarray] = None
        self.active_frame_count: int = 0
        self.settled_frame_count: int = 0

    def reset(self) -> None:
        """Resets segmentation state machine and temporal buffers."""
        self.state = GestureState.IDLE
        self.smoothed_energy = 0.0
        self.prev_pts = None
        self.active_frame_count = 0
        self.settled_frame_count = 0

    def process_frame(
        self,
        hand_pts_21x3: Optional[np.ndarray],
        dt: float = 1.0 / 30.0,
    ) -> Tuple[GestureState, float, bool]:
        """
        Processes a single frame's hand landmark coordinates.

        Returns:
            (current_state, smoothed_kinetic_energy, should_trigger_inference)
        """
        if hand_pts_21x3 is None or len(hand_pts_21x3) != 21:
            self.reset()
            return GestureState.IDLE, 0.0, False

        curr_pts = np.asarray(hand_pts_21x3, dtype=np.float32)

        # Guard against zero/missing hand
        if np.all(curr_pts == 0.0):
            self.reset()
            return GestureState.IDLE, 0.0, False

        # Compute raw kinetic energy
        if self.prev_pts is not None and self.prev_pts.shape == curr_pts.shape:
            # Time-normalized velocity vectors
            dt_clamped = max(dt, 1e-4)
            time_scale = (1.0 / 30.0) / dt_clamped
            diff_vecs = (curr_pts - self.prev_pts) * time_scale

            # Weight fingertips and wrist more heavily as primary articulators
            # Wrist: 0, Thumb tip: 4, Index tip: 8, Middle tip: 12, Ring tip: 16, Pinky tip: 20
            weights = np.ones(21, dtype=np.float32)
            weights[[0, 4, 8, 12, 16, 20]] = 2.0
            norms = np.linalg.norm(diff_vecs, axis=-1) * weights
            raw_energy = float(np.sum(norms) / np.sum(weights))
        else:
            raw_energy = 0.0

        self.prev_pts = curr_pts.copy()

        # Update smoothed kinetic energy with EMA
        self.smoothed_energy = self.alpha_smooth * raw_energy + (1.0 - self.alpha_smooth) * self.smoothed_energy

        should_trigger = False

        # State transitions
        if self.state == GestureState.IDLE:
            if self.smoothed_energy >= self.energy_threshold:
                self.state = GestureState.GESTURE_START
                self.active_frame_count = 1
                self.settled_frame_count = 0

        elif self.state in (GestureState.GESTURE_START, GestureState.GESTURE_ACTIVE):
            self.active_frame_count += 1

            if self.smoothed_energy < self.energy_threshold:
                self.settled_frame_count += 1
                if self.settled_frame_count >= self.settle_window_frames:
                    # Motion has settled! Check if gesture duration was valid
                    if self.active_frame_count >= self.min_gesture_frames:
                        self.state = GestureState.GESTURE_SETTLED
                        should_trigger = True
                    else:
                        # Too brief to be an intentional sign (transient twitch)
                        self.state = GestureState.IDLE
            else:
                self.state = GestureState.GESTURE_ACTIVE
                self.settled_frame_count = 0

            # Guard against gesture buffer overflow
            if self.active_frame_count >= self.max_gesture_frames:
                self.state = GestureState.GESTURE_SETTLED
                should_trigger = True

        elif self.state == GestureState.GESTURE_SETTLED:
            # Transition back to IDLE after triggering
            if self.smoothed_energy < self.energy_threshold:
                self.state = GestureState.IDLE
            else:
                self.state = GestureState.GESTURE_START
                self.active_frame_count = 1
                self.settled_frame_count = 0

        return self.state, self.smoothed_energy, should_trigger
