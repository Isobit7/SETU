"""
One-Euro Filter implementation for real-time 3D landmark jitter elimination.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '12), 2527-2530.
"""

import math
import time
from typing import Optional, Union
import numpy as np


class LowPassFilter:
    """Standard first-order exponential smoothing low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha: float = alpha
        self.s: Optional[np.ndarray] = None

    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = np.array(value, dtype=np.float32, copy=True)
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self) -> None:
        self.s = None


class OneEuroFilter:
    """
    Adaptive low-pass filter that eliminates high-frequency jitter during stationary
    hand poses while dynamically adapting to prevent lag during rapid hand gestures.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower values reduce jitter when still.
            beta: Speed coefficient. Higher values reduce lag during fast movements.
            d_cutoff: Cutoff frequency for the derivative filter (Hz).
        """
        self.min_cutoff: float = float(min_cutoff)
        self.beta: float = float(beta)
        self.d_cutoff: float = float(d_cutoff)

        self.x_filter: LowPassFilter = LowPassFilter()
        self.dx_filter: LowPassFilter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: Union[float, np.ndarray], dt: float) -> Union[float, np.ndarray]:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filter input landmark array (any arbitrary shape, e.g., (21, 3) or (126,)).
        """
        x_arr = np.asarray(x, dtype=np.float32)

        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filter.filter(x_arr, alpha=1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp

        # Guard against degenerate dt (e.g. back-to-back calls in sub-microsecond)
        if dt <= 1e-5:
            dt = 1e-4

        # Compute derivative (rate of change / speed)
        prev_s = self.x_filter.s
        if prev_s is None:
            dx = np.zeros_like(x_arr)
        else:
            dx = (x_arr - prev_s) / dt

        # Filter the derivative to get smooth speed estimate
        edx = self.dx_filter.filter(dx, alpha=self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff frequency: increases when hand moves fast, decreases when stationary
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)

        # Filter the signal with the adaptive alpha
        return self.x_filter.filter(x_arr, alpha=alpha)

    def reset(self) -> None:
        """Reset filter history (e.g. when hand leaves frame)."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None

