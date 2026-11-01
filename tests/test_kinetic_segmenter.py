"""
Unit tests for KineticEnergySegmenter.
Verifies idle, start, active, and settled gesture states.
"""

import numpy as np
from backend.app.services.vision.kinetic_segmenter import KineticEnergySegmenter, GestureState


def test_kinetic_segmenter_idle_when_still():
    segmenter = KineticEnergySegmenter(energy_threshold=0.03)
    hand = np.ones((21, 3), dtype=np.float32) * 0.5
    
    # Still hand for 10 frames
    for _ in range(10):
        state, energy, trigger = segmenter.process_frame(hand)
        assert state == GestureState.IDLE
        assert trigger is False
        assert energy < 0.01


def test_kinetic_segmenter_gesture_lifecycle():
    segmenter = KineticEnergySegmenter(
        energy_threshold=0.03,
        settle_window_frames=2,
        min_gesture_frames=4,
    )
    
    # 1. Start still
    base_hand = np.ones((21, 3), dtype=np.float32) * 0.5
    segmenter.process_frame(base_hand)
    
    # 2. Fast movement for 6 frames (signing "HELLO")
    for i in range(1, 7):
        moving_hand = base_hand + float(i * 0.08)
        state, energy, trigger = segmenter.process_frame(moving_hand)
        assert state in (GestureState.GESTURE_START, GestureState.GESTURE_ACTIVE)
        assert trigger is False
        assert energy > 0.02
        
    # 3. Settle still
    settled_hand = moving_hand.copy()
    triggered = False
    for _ in range(6):
        state, energy, trig = segmenter.process_frame(settled_hand)
        if trig:
            triggered = True
            assert state == GestureState.GESTURE_SETTLED
            break

    assert triggered is True
    
    # 4. Next frame returns to IDLE
    state, energy, trigger3 = segmenter.process_frame(settled_hand)
    assert state == GestureState.IDLE
    assert trigger3 is False


def test_kinetic_segmenter_ignores_twitch():
    segmenter = KineticEnergySegmenter(
        energy_threshold=0.03,
        settle_window_frames=2,
        min_gesture_frames=5,  # Requires at least 5 frames
    )
    
    base_hand = np.zeros((21, 3), dtype=np.float32)
    segmenter.process_frame(base_hand)
    
    # Single frame twitch
    twitch_hand = base_hand + 0.10
    segmenter.process_frame(twitch_hand)
    
    # Immediately still
    segmenter.process_frame(twitch_hand)
    state, _, trigger = segmenter.process_frame(twitch_hand)
    
    # Should not trigger because active duration (1 frame) < min_gesture_frames (5 frames)
    assert trigger is False
