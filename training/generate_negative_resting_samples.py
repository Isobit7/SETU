"""
SETU - Authentic NO_SIGN Sample Generator
Generates realistic non-signing sequences (empty frames, resting hands, face touching,
and edge transitions) to train the model to reject false positive activations.
"""

from typing import List, Optional
import numpy as np

from training.feature_extractor import (
    FEATURE_DIM,
    validate_schema_vector,
    FeatureExtractor158,
)

class NoSignGenerator:
    """
    Produces genuine background and non-signing 158-D feature sequences.
    Enforces strict compliance with the immutable 158-feature schema.
    """
    def __init__(self, sequence_length: int = 30) -> None:
        self.sequence_length = sequence_length

    def generate_empty_sequence(self) -> np.ndarray:
        """Scenario 1: Signer is seated without hands raised (0 hands present)."""
        seq = np.zeros((self.sequence_length, FEATURE_DIM), dtype=np.float32)
        for t in range(self.sequence_length):
            validate_schema_vector(seq[t])
        return seq

    def generate_resting_hands_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Scenario 2: Hands resting on lap or desk, low in frame (y > 0.85) with natural tremor.
        """
        rng = np.random.default_rng(seed)
        seq = np.zeros((self.sequence_length, FEATURE_DIM), dtype=np.float32)

        # Baseline resting hand coordinates
        rh_base = np.zeros((21, 3), dtype=np.float32)
        lh_base = np.zeros((21, 3), dtype=np.float32)
        
        # Place wrists low and separated
        rh_base[0] = [0.65, 0.90, 0.0]
        lh_base[0] = [0.35, 0.90, 0.0]
        for i in range(1, 21):
            rh_base[i] = rh_base[0] + rng.normal(0, 0.02, size=3).astype(np.float32)
            lh_base[i] = lh_base[0] + rng.normal(0, 0.02, size=3).astype(np.float32)

        for t in range(self.sequence_length):
            # Add minor frame-to-frame micro-drift
            noise_r = rng.normal(0, 0.002, size=(21, 3)).astype(np.float32)
            noise_l = rng.normal(0, 0.002, size=(21, 3)).astype(np.float32)
            
            vec = FeatureExtractor158.extract_from_hands(
                right_hand=rh_base + noise_r,
                left_hand=lh_base + noise_l
            )
            seq[t] = vec

        return seq

    def generate_face_touch_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Scenario 3: Single hand resting on chin/cheek (common idle non-signing pose).
        """
        rng = np.random.default_rng(seed)
        seq = np.zeros((self.sequence_length, FEATURE_DIM), dtype=np.float32)

        hand_base = np.zeros((21, 3), dtype=np.float32)
        hand_base[0] = [0.55, 0.40, -0.05] # Near chin
        for i in range(1, 21):
            hand_base[i] = hand_base[0] + rng.normal(0, 0.02, size=3).astype(np.float32)

        for t in range(self.sequence_length):
            drift = rng.normal(0, 0.001, size=(21, 3)).astype(np.float32)
            # Only right hand present; left hand idle/untracked
            vec = FeatureExtractor158.extract_from_hands(
                right_hand=hand_base + drift,
                left_hand=None
            )
            seq[t] = vec

        return seq

    def generate_inter_sign_transition_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Scenario 4: Rapid transitional movement between signs (co-articulation transitions).
        Hands moving linearly across the frame without forming held phonemes.
        """
        rng = np.random.default_rng(seed)
        seq = np.zeros((self.sequence_length, FEATURE_DIM), dtype=np.float32)

        start_r = np.array([0.45, 0.50, 0.0], dtype=np.float32)
        end_r = np.array([0.70, 0.85, 0.05], dtype=np.float32)

        for t in range(self.sequence_length):
            alpha = t / max(1, self.sequence_length - 1)
            wrist_pos = (1.0 - alpha) * start_r + alpha * end_r
            rh = np.zeros((21, 3), dtype=np.float32)
            rh[0] = wrist_pos
            for i in range(1, 21):
                rh[i] = rh[0] + rng.normal(0, 0.025, size=3).astype(np.float32)

            vec = FeatureExtractor158.extract_from_hands(
                right_hand=rh,
                left_hand=None,
            )
            seq[t] = vec

        return seq

    def generate_ambient_gesture_sequence(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Scenario 5: Ambient non-signing gestures (adjusting glasses, collar, fidgeting).
        """
        rng = np.random.default_rng(seed)
        seq = np.zeros((self.sequence_length, FEATURE_DIM), dtype=np.float32)

        rh_head = np.array([0.52, 0.28, -0.05], dtype=np.float32)
        for t in range(self.sequence_length):
            rh = np.zeros((21, 3), dtype=np.float32)
            jitter = rng.normal(0, 0.015, size=3).astype(np.float32)
            rh[0] = rh_head + jitter
            for i in range(1, 21):
                rh[i] = rh[0] + rng.normal(0, 0.02, size=3).astype(np.float32)

            vec = FeatureExtractor158.extract_from_hands(
                right_hand=rh,
                left_hand=None,
            )
            seq[t] = vec

        return seq

    def generate_batch(self, count_per_scenario: int = 5, base_seed: int = 42) -> List[np.ndarray]:
        """Generates a balanced batch of NO_SIGN sequences across diverse scenarios."""
        samples = []
        for i in range(count_per_scenario):
            samples.append(self.generate_empty_sequence())
            samples.append(self.generate_resting_hands_sequence(seed=base_seed + i * 5))
            samples.append(self.generate_face_touch_sequence(seed=base_seed + i * 5 + 1))
            samples.append(self.generate_inter_sign_transition_sequence(seed=base_seed + i * 5 + 2))
            samples.append(self.generate_ambient_gesture_sequence(seed=base_seed + i * 5 + 3))
        return samples

