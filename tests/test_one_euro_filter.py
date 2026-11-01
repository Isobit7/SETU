"""
Unit tests for OneEuroFilter implementation.
Verifies noise attenuation when stationary and low latency during rapid motion.
"""

import numpy as np
from backend.app.services.vision.one_euro_filter import OneEuroFilter, LowPassFilter


def test_low_pass_filter():
    lpf = LowPassFilter(alpha=0.5)
    assert lpf.s is None
    
    val1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    res1 = lpf.filter(val1)
    np.testing.assert_allclose(res1, val1)
    
    val2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    res2 = lpf.filter(val2)
    # alpha=0.5 -> 0.5 * [2,4,6] + 0.5 * [1,2,3] = [1.5, 3.0, 4.5]
    np.testing.assert_allclose(res2, np.array([1.5, 3.0, 4.5], dtype=np.float32))
    
    lpf.reset()
    assert lpf.s is None


def test_one_euro_filter_stationary_noise_reduction():
    """Verify that stationary landmarks with high-frequency Gaussian noise are strongly smoothed."""
    filter = OneEuroFilter(min_cutoff=0.8, beta=0.01)
    
    # Ground truth stationary point
    true_pt = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    # Add Gaussian jitter
    np.random.seed(42)
    raw_jitter = []
    filtered_pts = []
    
    t = 0.0
    dt = 1.0 / 30.0  # 30 FPS
    
    for _ in range(60):
        noise = np.random.normal(0, 0.05, size=(3,)).astype(np.float32)
        noisy_pt = true_pt + noise
        raw_jitter.append(noisy_pt)
        
        filtered = filter.filter(noisy_pt, timestamp=t)
        filtered_pts.append(filtered)
        t += dt
        
    raw_var = np.var(raw_jitter, axis=0)
    filtered_var = np.var(filtered_pts[10:], axis=0)  # skip warm-up
    
    # Filtered variance should be significantly smaller than raw jitter variance
    assert np.all(filtered_var < raw_var * 0.4)


def test_one_euro_filter_fast_motion_responsiveness():
    """Verify that fast motion quickly increases cutoff frequency to avoid dragging behind."""
    filter = OneEuroFilter(min_cutoff=0.8, beta=0.05)
    
    t = 0.0
    dt = 1.0 / 30.0
    
    # Still at origin
    for _ in range(10):
        filter.filter(np.array([0.0, 0.0, 0.0], dtype=np.float32), timestamp=t)
        t += dt
        
    # Sudden jump to (1.0, 1.0, 1.0)
    jump_pt = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(5):
        res = filter.filter(jump_pt, timestamp=t)
        t += dt

    # Within 5 frames (~160ms), the adaptive filter rapidly catches up (> 0.65)
    assert np.all(res > 0.65)
    
    filter.reset()
    assert filter.last_time is None
