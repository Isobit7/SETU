"""
SETU — WebSocket Real-Time ISL Recognition Handler
Connects client video streams with MediaPipe, kinetic energy gesture segmentation,
confidence hysteresis, 4-frame streak debouncers, and SentenceAccumulator.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.services.pipeline_manager import pipeline
from backend.app.services.vision.landmarks import HolisticLandmarkerResult
from training.feature_extractor import (
    FeatureExtractor158,
    extract_facial_expression_features,
    extract_realsign_hand_features,
)
from training.features.geometry import extract_dual_hand_126d, extract_body_anchored_features
from models.adapters.realsign import RealSignPrediction
from models.adapters.sequence_lstm import SequencePrediction as ISLSequencePrediction
from models.adapters.mvp import MVPClassifierPrediction as ISLPrediction
from models.adapters.include import StreamingIncludeAdapter, INCLUDEResult
from backend.app.services.nlp import SentenceAccumulator
from config.settings import (
    CONFIDENCE_THRESHOLD_ALPHABET,
    CONFIDENCE_THRESHOLD_SEQUENCE,
    STREAK_ALPHABET,
    STREAK_SEQUENCE,
)

logger = logging.getLogger("setu.backend.ws")

DISPLAY_NAMES: Dict[str, str] = {
    "hello": "HELLO",
    "thank_you": "THANK YOU",
    "good_morning": "GOOD MORNING",
    "i": "I (ME)",
    "you": "YOU",
    "namaste": "NAMASTE",
    "no_sign": "Waiting for a sign...",
    "unknown": "UNKNOWN / UNCLEAR",
}

FACE_KEYPOINT_INDICES = [
    # Face oval
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Lips outer & inner
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # Left eye
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Right eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Nose
    1, 2, 98, 327,
]


def serialize_landmarks(raw_lms: Any, sample_indices: Optional[List[int]] = None) -> List[Dict[str, float]]:
    """Serializes MediaPipe NormalizedLandmark list into a clean dictionary list."""
    if raw_lms is None:
        return []
    if hasattr(raw_lms, "landmark"):
        first_list = raw_lms.landmark
    elif isinstance(raw_lms, list) and len(raw_lms) > 0 and isinstance(raw_lms[0], list):
        first_list = raw_lms[0]
    elif hasattr(raw_lms, "__len__") and len(raw_lms) == 0:
        return []
    else:
        first_list = raw_lms

    out: List[Dict[str, float]] = []
    if sample_indices:
        for idx in sample_indices:
            if idx < len(first_list):
                lm = first_list[idx]
                out.append({"x": round(float(lm.x), 4), "y": round(float(lm.y), 4), "z": round(float(getattr(lm, "z", 0.0)), 4)})
    else:
        for lm in first_list:
            out.append({"x": round(float(lm.x), 4), "y": round(float(lm.y), 4), "z": round(float(getattr(lm, "z", 0.0)), 4)})
    return out


def check_subtitle_collision(left_hand: List[Dict[str, float]], right_hand: List[Dict[str, float]]) -> bool:
    """Checks if hands are in the lower central zone where movie subtitles sit."""
    for pt in left_hand + right_hand:
        if pt["y"] >= 0.72 and 0.20 <= pt["x"] <= 0.80:
            return True
    return False


def get_positioning_guidance(
    pose_pts: List[Dict[str, float]],
    has_hands: bool,
    lighting_score: Optional[float] = None,
) -> str:
    """Generates real positioning guidance based on upper body keypoints and ambient lighting."""
    if not pose_pts or len(pose_pts) < 13:
        return "Step into frame so your head and upper body are visible"

    if lighting_score is not None and lighting_score < 0.38:
        return "Low light detected — please turn on a light facing you"

    nose = pose_pts[0]
    left_shoulder = pose_pts[11]
    right_shoulder = pose_pts[12]

    shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])

    if shoulder_width > 0.65:
        return "Too close to camera — please step back slightly (ideal: 50–70 cm)"
    if shoulder_width < 0.16:
        return "Too far from camera — please move closer (ideal: 50–70 cm)"

    if nose["y"] < 0.05:
        return "Camera too low — tilt up or lower your position"
    if nose["y"] > 0.45:
        return "Camera too high — step back or raise camera"

    if not has_hands:
        return "Raise your hands into camera view to sign"

    return "Position good — Ready to translate"


async def websocket_recognize(
    websocket: WebSocket,
    model: Optional[str] = None,
    threshold: Optional[float] = None,
) -> None:
    """Low-latency WebSocket endpoint for live camera frames and streaming predictions."""
    await websocket.accept()
    if not pipeline.is_ready:
        try:
            pipeline.load()
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Pipeline initialization failed: {e}",
            })
            await websocket.close()
            return

    # Per-connection state
    active_model = (model or "realsign").lower()
    if active_model == "include":
        active_model = "include_stream"
    if active_model not in ("realsign", "isl_mvp", "isl_sequence_lstm", "include_stream"):
        active_model = "realsign"

    streaming_include: Optional[StreamingIncludeAdapter] = None
    if pipeline.include_service is not None:
        streaming_include = StreamingIncludeAdapter(service=pipeline.include_service)

    rolling_features: deque = deque(maxlen=15)
    client_threshold: Optional[float] = None
    if threshold is not None:
        try:
            client_threshold = max(0.10, min(0.95, float(threshold)))
        except (ValueError, TypeError):
            client_threshold = None
    frame_count = 0
    total_stream_frames = 0
    last_fps_calc = time.time()
    fps = 0.0

    # Temporal Debouncing & Confidence Hysteresis State
    candidate_label: str = ""
    candidate_streak: int = 0
    last_committed_sign: str = ""
    last_committed_time: float = 0.0
    had_idle_baseline: bool = True

    # RealSign Alphabet 4-Frame Streak Debouncer State
    alphabet_streak_candidate: str = ""
    alphabet_streak_count: int = 0
    alphabet_last_committed: str = ""
    alphabet_last_committed_time: float = 0.0
    alphabet_had_baseline: bool = True
    alphabet_repeat_lock: bool = False
    last_hand_seen_time: float = time.time()

    # Word / Phrase Persistence & Latching State
    last_word_label: str = ""
    last_word_display: str = ""
    last_word_conf: float = 0.0
    last_word_is_known: bool = False
    last_word_probs: Dict[str, float] = {}
    last_word_top3: List[Tuple[str, float]] = []
    last_word_pred_time: float = 0.0

    # Motion-Based Kinetic Energy & Active Gesture Segmentation State
    prev_hand_pts: Optional[np.ndarray] = None
    prev_frame_perf_time: float = time.perf_counter()
    gesture_sequence_buffer: List[np.ndarray] = []
    is_actively_signing: bool = False
    settled_frames_count: int = 0
    KINETIC_ENERGY_THRESHOLD: float = 0.025

    # Multimodal Object Detection & Show-and-Sign State
    current_detected_objects: List[Any] = []
    fused_sentence: Optional[str] = None
    lighting_score: float = 1.0
    was_low_light_enhanced: bool = False

    # Disambiguation HUD & Low-Margin Gate State
    disambiguation_active: bool = False
    disambiguation_top1: str = ""
    disambiguation_top2: str = ""
    disambiguation_options: List[Dict[str, Any]] = []
    disambiguation_start_time: float = 0.0

    # Real-Time Sentence Accumulator & ISL Gloss Parser
    accumulator = SentenceAccumulator()

    logger.info("Client connected to SETU recognition WebSocket. Active model: %s", active_model)
    loop = asyncio.get_running_loop()

    async def _process_control_text(text_str: str) -> None:
        nonlocal active_model, client_threshold, is_actively_signing, settled_frames_count
        nonlocal prev_hand_pts, candidate_label, candidate_streak, last_committed_sign
        nonlocal last_committed_time, had_idle_baseline
        nonlocal alphabet_streak_candidate, alphabet_streak_count, alphabet_last_committed
        nonlocal last_word_label, last_word_display, last_word_conf, last_word_is_known
        nonlocal last_word_probs, last_word_top3, last_word_pred_time
        nonlocal disambiguation_active, disambiguation_top1, disambiguation_top2, disambiguation_options
        try:
            payload = json.loads(text_str)
            action = payload.get("action")
            if action == "reset":
                rolling_features.clear()
                gesture_sequence_buffer.clear()
                accumulator.reset()
                is_actively_signing = False
                settled_frames_count = 0
                prev_hand_pts = None
                candidate_label = ""
                candidate_streak = 0
                last_committed_sign = ""
                last_committed_time = 0.0
                had_idle_baseline = True
                alphabet_streak_candidate = ""
                alphabet_streak_count = 0
                alphabet_last_committed = ""
                alphabet_last_committed_time = 0.0
                alphabet_had_baseline = True
                last_word_label = ""
                last_word_display = ""
                last_word_conf = 0.0
                last_word_is_known = False
                last_word_probs.clear()
                last_word_top3.clear()
                last_word_pred_time = 0.0
                if pipeline.landmarker is not None:
                    pipeline.landmarker.reset_smoothing()
                if pipeline.hand_landmarker is not None:
                    pipeline.hand_landmarker.reset_smoothing()
                if pipeline.realsign_classifier is not None:
                    pipeline.realsign_classifier.reset_smoothing()
                await websocket.send_json({"type": "info", "message": "Buffer reset"})
            elif action == "clear_sentence":
                accumulator.reset()
                await websocket.send_json({
                    "type": "info",
                    "message": "Sentence buffer cleared",
                    "sentence_buffer": accumulator.sentence_buffer,
                    "translated_text": accumulator.translated_text,
                })
            elif action == "backspace":
                accumulator.backspace()
                await websocket.send_json({
                    "type": "info",
                    "message": "Backspace applied",
                    "sentence_buffer": accumulator.sentence_buffer,
                    "translated_text": accumulator.translated_text,
                })
            elif action in ("resolve_disambiguation", "select_disambiguation"):
                selected_label = str(payload.get("selected", payload.get("choice", ""))).strip().lower()
                if selected_label in ("left", "option_a") and disambiguation_top1:
                    selected_label = disambiguation_top1.lower()
                elif selected_label in ("right", "option_b") and disambiguation_top2:
                    selected_label = disambiguation_top2.lower()

                if selected_label:
                    is_letter = (len(selected_label) == 1 and selected_label.isalpha())
                    disp = selected_label.upper() if is_letter else DISPLAY_NAMES.get(selected_label, selected_label.upper().replace("_", " "))
                    token_type = "letter" if is_letter else "phrase"
                    accumulator.add_token(disp, token_type=token_type, timestamp=time.time())
                    disambiguation_active = False
                    disambiguation_options = []
                    disambiguation_top1 = ""
                    disambiguation_top2 = ""
                    await websocket.send_json({
                        "type": "info",
                        "message": f"Resolved disambiguation with {disp}",
                        "sentence_buffer": accumulator.sentence_buffer,
                    })
                else:
                    await websocket.send_json({
                        "type": "info",
                        "message": "Disambiguation acknowledged",
                    })
            elif action == "accept_prediction":
                phrase = str(payload.get("phrase", "")).strip()
                if phrase:
                    accumulator.add_token(phrase.upper(), token_type="phrase", timestamp=time.time())
                    await websocket.send_json({
                        "type": "info",
                        "message": f"Accepted predictive phrase: {phrase}",
                        "sentence_buffer": accumulator.sentence_buffer,
                    })
            elif action == "set_model":
                req_model = str(payload.get("model", "realsign")).lower()
                if req_model == "include":
                    req_model = "include_stream"
                if req_model in ("realsign", "isl_mvp", "isl_sequence_lstm", "include_stream"):
                    active_model = req_model
                    rolling_features.clear()
                    gesture_sequence_buffer.clear()
                    if streaming_include is not None:
                        streaming_include.reset()
                    accumulator.reset()
                    is_actively_signing = False
                    settled_frames_count = 0
                    prev_hand_pts = None
                    candidate_label = ""
                    candidate_streak = 0
                    last_committed_sign = ""
                    last_committed_time = 0.0
                    had_idle_baseline = True
                    alphabet_streak_candidate = ""
                    alphabet_streak_count = 0
                    alphabet_last_committed = ""
                    last_word_label = ""
                    last_word_display = ""
                    last_word_conf = 0.0
                    last_word_is_known = False
                    last_word_probs.clear()
                    last_word_top3.clear()
                    last_word_pred_time = 0.0
                    if pipeline.landmarker is not None:
                        pipeline.landmarker.reset_smoothing()
                    if pipeline.hand_landmarker is not None:
                        pipeline.hand_landmarker.reset_smoothing()
                    if pipeline.realsign_classifier is not None:
                        pipeline.realsign_classifier.reset_smoothing()
                    await websocket.send_json({
                        "type": "info",
                        "message": f"Active classifier model switched to {active_model}",
                        "active_model": active_model,
                    })
            elif action == "set_threshold":
                val = float(payload.get("threshold", CONFIDENCE_THRESHOLD_ALPHABET))
                client_threshold = max(0.10, min(0.95, val))
                await websocket.send_json({
                    "type": "info",
                    "message": f"Confidence threshold set to {client_threshold}",
                    "threshold": client_threshold,
                })
        except Exception as e:
            logger.warning("Error processing text command: %s", e)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # WebSocket Backpressure Guard: Drain queued stale frames
                while True:
                    try:
                        next_msg = await asyncio.wait_for(websocket.receive(), timeout=0.0002)
                        if "bytes" in next_msg and next_msg["bytes"]:
                            message = next_msg
                        elif "text" in next_msg and next_msg["text"]:
                            await _process_control_text(next_msg["text"])
                    except (asyncio.TimeoutError, TimeoutError):
                        break

                t0 = time.perf_counter()
                frame_bytes = message["bytes"]

                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = await loop.run_in_executor(None, cv2.imdecode, np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Failed to decode JPEG frame"})
                    continue

                # Adaptive low-light CLAHE enhancement
                lighting_score = 1.0
                was_low_light_enhanced = False
                if pipeline.enhancer is not None:
                    frame, lighting_score, was_low_light_enhanced = pipeline.enhancer.process(frame)

                frame_count += 1
                total_stream_frames += 1
                now = time.time()
                if now - last_fps_calc >= 1.0:
                    fps = round(frame_count / (now - last_fps_calc), 1)
                    frame_count = 0
                    last_fps_calc = now

                confidence_threshold = client_threshold if client_threshold is not None else (
                    CONFIDENCE_THRESHOLD_ALPHABET if active_model == "realsign" else CONFIDENCE_THRESHOLD_SEQUENCE
                )

                def _extract_landmarks_sync(img: np.ndarray):
                    h_raw = None
                    hl_res = None

                    if active_model == "realsign":
                        # Fast-path for alphabet fingerspelling:
                        # HandLandmarker detects up to 2 hands with high precision and low latency (~20ms).
                        if pipeline.hand_landmarker is not None:
                            try:
                                hld_res = pipeline.hand_landmarker.detect(img)
                                if hld_res and getattr(hld_res, "hand_landmarks", None) and len(hld_res.hand_landmarks) > 0:
                                    h_raw = hld_res
                            except Exception as e:
                                logger.debug("HandLandmarker error: %s", e)

                        # Only fallback to heavyweight Holistic if HandLandmarker found no hands
                        if h_raw is None and pipeline.landmarker is not None:
                            hl_res = pipeline.landmarker.detect(img)
                        else:
                            hl_res = HolisticLandmarkerResult()
                    else:
                        # Sequence / Word models: need pose and face for 164-D vectors
                        if pipeline.landmarker is not None:
                            hl_res = pipeline.landmarker.detect(img)
                        else:
                            hl_res = HolisticLandmarkerResult()

                        # Only fallback to HandLandmarker if Holistic detected neither left nor right hand
                        if not (hl_res.left_hand_landmarks or hl_res.right_hand_landmarks) and pipeline.hand_landmarker is not None:
                            try:
                                hld_res = pipeline.hand_landmarker.detect(img)
                                if hld_res and getattr(hld_res, "hand_landmarks", None) and len(hld_res.hand_landmarks) > 0:
                                    h_raw = hld_res
                            except Exception as e:
                                logger.debug("HandLandmarker secondary error: %s", e)

                    return h_raw, hl_res

                try:
                    hand_landmarks_raw, hl_result = await loop.run_in_executor(
                        None, _extract_landmarks_sync, frame
                    )
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Landmark detection failed: {e}",
                    })
                    continue

                left_hand = serialize_landmarks(hl_result.left_hand_landmarks)
                right_hand = serialize_landmarks(hl_result.right_hand_landmarks)
                pose = serialize_landmarks(hl_result.pose_landmarks)
                face = serialize_landmarks(hl_result.face_landmarks, FACE_KEYPOINT_INDICES)

                if hand_landmarks_raw is not None:
                    if hasattr(hand_landmarks_raw, "hand_landmarks") and hand_landmarks_raw.hand_landmarks:
                        for i, h_lm in enumerate(hand_landmarks_raw.hand_landmarks):
                            h_lbl = "right"
                            if hasattr(hand_landmarks_raw, "handedness") and i < len(hand_landmarks_raw.handedness) and hand_landmarks_raw.handedness[i]:
                                h_lbl = getattr(hand_landmarks_raw.handedness[i][0], "category_name", "right").lower()
                            s_lm = serialize_landmarks(h_lm)
                            if h_lbl == "left":
                                if len(left_hand) == 0:
                                    left_hand = s_lm
                            else:
                                if len(right_hand) == 0:
                                    right_hand = s_lm
                    else:
                        hand_pts_serialized = serialize_landmarks(hand_landmarks_raw)
                        if len(right_hand) == 0:
                            right_hand = hand_pts_serialized

                has_left = len(left_hand) > 0
                has_right = len(right_hand) > 0
                hand_detected = has_left or has_right or (hand_landmarks_raw is not None)

                now = time.time()
                if hand_detected:
                    last_hand_seen_time = now
                    hand_absent_duration = 0.0
                else:
                    hand_absent_duration = now - last_hand_seen_time

                # Reset repeat lock and baseline when user relaxes/drops hands for >= 0.45s
                if hand_absent_duration >= 0.45:
                    alphabet_repeat_lock = False
                    alphabet_last_committed = ""
                    had_idle_baseline = True

                # Periodic COCO Object Detection for Show-and-Sign interaction (every 10 frames)
                if pipeline.object_detector is not None and (total_stream_frames % 10 == 0):
                    try:
                        detected_objs = await loop.run_in_executor(
                            None,
                            pipeline.object_detector.detect,
                            frame,
                            0.65,
                            left_hand + right_hand,
                        )
                        current_detected_objects = detected_objs
                    except Exception as e:
                        logger.debug("Object detection error: %s", e)

                # Robust 176-D body-anchored feature extraction
                face_lms = getattr(hl_result, "face_landmarks", None)
                pose_lms = getattr(hl_result, "pose_landmarks", None)
                facial_ratios = extract_facial_expression_features(face_lms)
                body_anchors = extract_body_anchored_features(
                    pose_lms,
                    left_hand=left_hand if has_left else None,
                    right_hand=right_hand if has_right else None,
                )

                if has_left or has_right:
                    feat_158 = FeatureExtractor158.extract_from_hands(
                        right_hand=right_hand if has_right else None,
                        left_hand=left_hand if has_left else None,
                    )
                    feat_164 = np.concatenate([feat_158, facial_ratios]).astype(np.float32)
                else:
                    feat_164 = pipeline.extractor.extract_from_mediapipe_result(hl_result, include_facial=True)

                feat_176 = np.concatenate([feat_164, body_anchors]).astype(np.float32)
                rolling_features.append(feat_176)

                if active_model == "include_stream" and streaming_include is not None:
                    streaming_include.append_frame(pose, left_hand, right_hand)

                hands_in_subtitle_zone = check_subtitle_collision(left_hand, right_hand)
                guidance_text = get_positioning_guidance(pose, hand_detected, lighting_score=lighting_score)

                curr_time_perf = time.perf_counter()
                dt_frame = max(curr_time_perf - prev_frame_perf_time, 0.001)
                prev_frame_perf_time = curr_time_perf

                def _extract_pt(lm):
                    if isinstance(lm, dict):
                        return [float(lm.get("x", 0.0)), float(lm.get("y", 0.0)), float(lm.get("z", 0.0))]
                    return [float(getattr(lm, "x", 0.0)), float(getattr(lm, "y", 0.0)), float(getattr(lm, "z", 0.0))]

                current_hand_pts_list = []
                is_left_only_hand = False
                if hand_landmarks_raw is not None:
                    if hasattr(hand_landmarks_raw, "hand_landmarks") and hand_landmarks_raw.hand_landmarks:
                        for h_lm in hand_landmarks_raw.hand_landmarks[:1]:
                            current_hand_pts_list.extend([_extract_pt(lm) for lm in h_lm])
                    else:
                        current_hand_pts_list.extend([_extract_pt(lm) for lm in hand_landmarks_raw])
                elif hl_result and getattr(hl_result, "right_hand_landmarks", None):
                    current_hand_pts_list.extend([_extract_pt(lm) for lm in hl_result.right_hand_landmarks])
                elif hl_result and getattr(hl_result, "left_hand_landmarks", None):
                    current_hand_pts_list.extend([_extract_pt(lm) for lm in hl_result.left_hand_landmarks])
                    is_left_only_hand = True

                if is_left_only_hand and len(current_hand_pts_list) == 21:
                    # Mirror X coordinate so geometric/heuristic checks evaluate canonical hand
                    for pt_item in current_hand_pts_list:
                        pt_item[0] = -pt_item[0]

                kinetic_energy = 0.0
                if current_hand_pts_list:
                    curr_pts_arr = np.array(current_hand_pts_list, dtype=np.float32)
                    if prev_hand_pts is not None and prev_hand_pts.shape == curr_pts_arr.shape:
                        time_scale = (1.0 / 30.0) / dt_frame
                        diff_vecs = (curr_pts_arr - prev_hand_pts) * time_scale
                        kinetic_energy = float(np.sum(np.linalg.norm(diff_vecs, axis=-1)))
                    prev_hand_pts = curr_pts_arr
                else:
                    prev_hand_pts = None

                hand_source = None
                if hand_landmarks_raw is not None:
                    hand_source = hand_landmarks_raw
                elif hl_result and (getattr(hl_result, "right_hand_landmarks", None) or getattr(hl_result, "left_hand_landmarks", None)):
                    hand_source = hl_result
                elif len(right_hand) > 0:
                    hand_source = right_hand
                elif len(left_hand) > 0:
                    hand_source = left_hand

                realsign_feat_63d: Optional[np.ndarray] = None
                realsign_feat_126d: Optional[np.ndarray] = None
                if hand_source is not None:
                    realsign_feat_63d = extract_realsign_hand_features(hand_source, dominant_hand="right")

                if has_left or has_right:
                    raw_lh = hl_result.left_hand_landmarks if (hl_result and getattr(hl_result, "left_hand_landmarks", None)) else (left_hand if has_left else None)
                    raw_rh = hl_result.right_hand_landmarks if (hl_result and getattr(hl_result, "right_hand_landmarks", None)) else (right_hand if has_right else None)
                    realsign_feat_126d = extract_dual_hand_126d(raw_lh, raw_rh)

                should_run_inference = False
                motion_settled_now = False

                if active_model == "realsign":
                    should_run_inference = (realsign_feat_126d is not None or realsign_feat_63d is not None)
                elif active_model == "include_stream":
                    should_run_inference = bool(streaming_include is not None and streaming_include.should_evaluate())
                elif active_model == "isl_mvp":
                    # ISL MVP model is ultra-lightweight (numpy/scikit ~50us). Run every frame hands are present
                    should_run_inference = bool(hand_detected and len(rolling_features) >= 5)
                elif active_model == "isl_sequence_lstm":
                    if kinetic_energy >= KINETIC_ENERGY_THRESHOLD:
                        is_actively_signing = True
                        settled_frames_count = 0
                        gesture_sequence_buffer.append(feat_176)
                        if len(gesture_sequence_buffer) > 60:
                            gesture_sequence_buffer.pop(0)
                    elif is_actively_signing:
                        settled_frames_count += 1
                        gesture_sequence_buffer.append(feat_176)
                        if settled_frames_count >= 1:
                            should_run_inference = True
                            motion_settled_now = True
                    else:
                        # Periodic streaming evaluation for held or steady signs (every 2 frames = 15 FPS evaluation)
                        if hand_detected and len(rolling_features) >= 8 and (total_stream_frames % 2 == 0):
                            should_run_inference = True

                realsign_pred: Optional[RealSignPrediction] = None
                seq_pred: Optional[ISLSequencePrediction] = None
                isl_pred: Optional[ISLPrediction] = None
                include_pred: Optional[INCLUDEResult] = None

                if should_run_inference:
                    segment_to_classify = list(gesture_sequence_buffer) if (is_actively_signing and len(gesture_sequence_buffer) >= 8) else list(rolling_features)
                    feat_array = np.array(segment_to_classify, dtype=np.float32)

                    if active_model == "include_stream" and streaming_include is not None:
                        include_pred = await loop.run_in_executor(
                            None,
                            streaming_include.predict_window,
                            confidence_threshold,
                        )

                    elif active_model == "realsign" and pipeline.realsign_classifier is not None:
                        target_realsign_feat = realsign_feat_126d if (realsign_feat_126d is not None and getattr(pipeline.realsign_classifier, "input_dim", 63) == 126) else realsign_feat_63d
                        if target_realsign_feat is None and realsign_feat_126d is not None:
                            target_realsign_feat = realsign_feat_126d
                        if target_realsign_feat is not None:
                            raw_pts_arr = np.array(current_hand_pts_list, dtype=np.float32) if len(current_hand_pts_list) == 21 else None
                            realsign_pred = await loop.run_in_executor(
                                None,
                                lambda: pipeline.realsign_classifier.predict(
                                    target_realsign_feat,
                                    confidence_threshold=confidence_threshold,
                                    apply_smoothing=True,
                                    apply_heuristic_boost=True,
                                    raw_landmarks_21x3=raw_pts_arr,
                                ),
                            )
                        else:
                            realsign_pred = pipeline.realsign_classifier._empty_prediction()
                            pipeline.realsign_classifier.reset_smoothing()

                    elif active_model == "isl_sequence_lstm" and pipeline.sequence_lstm_classifier is not None:
                        if len(segment_to_classify) >= 5:
                            seq_pred = await loop.run_in_executor(
                                None,
                                pipeline.sequence_lstm_classifier.predict_sequence,
                                feat_array,
                                confidence_threshold,
                            )
                        else:
                            seq_pred = pipeline.sequence_lstm_classifier._empty_prediction()

                    elif pipeline.classifier is not None:
                        if len(segment_to_classify) >= 5:
                            arr_mvp = feat_array
                            if arr_mvp.shape[0] < 30:
                                pad = np.repeat(arr_mvp[-1:], 30 - arr_mvp.shape[0], axis=0)
                                arr_mvp = np.concatenate([arr_mvp, pad], axis=0)
                            isl_pred = await loop.run_in_executor(
                                None,
                                pipeline.classifier.predict_features,
                                arr_mvp[:, :158],
                                confidence_threshold,
                            )
                        else:
                            isl_pred = ISLPrediction(
                                label="no_sign",
                                confidence=1.0,
                                is_known_sign=False,
                                probabilities={c: 0.0 for c in pipeline.classifier.class_names},
                                raw_class_idx=pipeline.classifier.class_names.index("no_sign"),
                            )

                    # Only reset active gesture buffer when gesture motion has settled
                    if motion_settled_now:
                        is_actively_signing = False
                        settled_frames_count = 0
                        gesture_sequence_buffer.clear()
                else:
                    if pipeline.realsign_classifier is not None:
                        realsign_pred = pipeline.realsign_classifier._empty_prediction()
                    if pipeline.sequence_lstm_classifier is not None:
                        seq_pred = pipeline.sequence_lstm_classifier._empty_prediction()
                    if pipeline.classifier is not None:
                        isl_pred = ISLPrediction(
                            label="no_sign",
                            confidence=1.0,
                            is_known_sign=False,
                            probabilities={c: 0.0 for c in pipeline.classifier.class_names},
                            raw_class_idx=pipeline.classifier.class_names.index("no_sign"),
                        )
                    if active_model == "include_stream" and streaming_include is not None:
                        include_pred = streaming_include.last_result or streaming_include._empty_result()

                t1 = time.perf_counter()
                latency_ms = round((t1 - t0) * 1000.0, 1)

                active_threshold = client_threshold if client_threshold is not None else (
                    CONFIDENCE_THRESHOLD_ALPHABET if active_model == "realsign" else CONFIDENCE_THRESHOLD_SEQUENCE
                )
                required_streak = STREAK_ALPHABET if active_model == "realsign" else STREAK_SEQUENCE

                has_realsign_feat = (realsign_feat_126d is not None) or (realsign_feat_63d is not None)

                if active_model == "realsign":
                    res_probs = {k: round(v, 4) for k, v in realsign_pred.probabilities.items()} if realsign_pred else {}
                    raw_conf = round(realsign_pred.confidence, 4) if realsign_pred else 0.0
                    raw_top_class = realsign_pred.top_3[0][0] if (realsign_pred and realsign_pred.top_3) else ("no_hand" if not hand_detected else "unknown")
                    raw_top_conf = float(realsign_pred.top_3[0][1]) if (realsign_pred and realsign_pred.top_3) else 0.0
                    raw_top_3 = realsign_pred.top_3 if realsign_pred else []
                    buffer_fill = 1.0 if (hand_detected and has_realsign_feat) else 0.0
                    is_pred_valid = bool(realsign_pred and realsign_pred.confidence >= active_threshold and realsign_pred.is_known_sign)
                    pred_label = realsign_pred.letter if realsign_pred else ""
                elif active_model == "include_stream":
                    res_probs = {c.label: round(c.confidence, 4) for c in include_pred.top5_candidates} if (include_pred and include_pred.top5_candidates) else {}
                    raw_conf = round(include_pred.confidence, 4) if include_pred else 0.0
                    raw_top_class = include_pred.label if include_pred else "awaiting_sign"
                    raw_top_conf = float(include_pred.confidence) if include_pred else 0.0
                    raw_top_3 = [(c.display_name, round(c.confidence, 4)) for c in include_pred.top5_candidates[:3]] if (include_pred and include_pred.top5_candidates) else []
                    buffer_fill = round(len(streaming_include.frame_buffer) / 30.0, 2) if streaming_include else 0.0
                    is_pred_valid = bool(include_pred and include_pred.is_known_sign and include_pred.confidence >= active_threshold and include_pred.label != "awaiting_sign")
                    pred_label = include_pred.label if include_pred else ""
                elif active_model == "isl_sequence_lstm":
                    res_probs = {k: round(v, 4) for k, v in seq_pred.probabilities.items()} if seq_pred else {}
                    raw_conf = round(seq_pred.confidence, 4) if seq_pred else 0.0
                    raw_top_class = seq_pred.top_3[0][0] if (seq_pred and seq_pred.top_3) else ("no_hand" if not hand_detected else "no_sign")
                    raw_top_conf = float(seq_pred.top_3[0][1]) if (seq_pred and seq_pred.top_3) else 0.0
                    raw_top_3 = seq_pred.top_3 if seq_pred else []
                    buffer_fill = round(len(gesture_sequence_buffer) / 15.0, 2) if is_actively_signing else round(len(rolling_features) / 15.0, 2)
                    is_pred_valid = bool(seq_pred and seq_pred.confidence >= active_threshold and seq_pred.is_known_sign and seq_pred.label != "no_sign")
                    pred_label = seq_pred.label if seq_pred else ""
                else:
                    res_probs = {k: round(v, 4) for k, v in isl_pred.probabilities.items()} if isl_pred else {}
                    raw_conf = round(isl_pred.confidence, 4) if isl_pred else 0.0
                    sorted_isl = sorted(isl_pred.probabilities.items(), key=lambda x: x[1], reverse=True)[:3] if isl_pred else []
                    raw_top_class = sorted_isl[0][0] if sorted_isl else ("no_hand" if not hand_detected else "no_sign")
                    raw_top_conf = float(sorted_isl[0][1]) if sorted_isl else 0.0
                    raw_top_3 = [(k, round(v, 4)) for k, v in sorted_isl]
                    buffer_fill = round(len(gesture_sequence_buffer) / 15.0, 2) if is_actively_signing else round(len(rolling_features) / 15.0, 2)
                    is_pred_valid = bool(isl_pred and isl_pred.confidence >= active_threshold and isl_pred.is_known_sign and isl_pred.label != "no_sign")
                    pred_label = isl_pred.label if isl_pred else ""

                if active_model == "realsign":
                    if not hand_detected or not has_realsign_feat or not is_pred_valid:
                        current_prediction = "Awaiting Sign Input"
                    else:
                        current_prediction = pred_label

                    if current_prediction == alphabet_streak_candidate:
                        alphabet_streak_count += 1
                    else:
                        alphabet_streak_candidate = current_prediction
                        alphabet_streak_count = 1

                    if alphabet_streak_count >= required_streak:
                        if alphabet_streak_candidate != "Awaiting Sign Input":
                            res_label = alphabet_streak_candidate
                            res_display = f"LETTER {alphabet_streak_candidate}"
                            res_conf = raw_conf
                            res_is_known = True
                            is_stable = True

                            # Strict Anti-Spam Refractory Lockout:
                            # 1. Once a letter commits, it is locked.
                            # 2. Holding the hand up will NEVER commit the letter a second time.
                            # 3. Only a DIFFERENT letter or relaxing hands for >= 0.45s allows a new commit.
                            can_commit = False
                            if alphabet_streak_candidate != alphabet_last_committed:
                                can_commit = True
                            elif not alphabet_repeat_lock:
                                can_commit = True

                            if can_commit and alphabet_streak_candidate not in ("", "no_sign", "unknown", "Awaiting Sign Input"):
                                accumulator.add_token(alphabet_streak_candidate, token_type="letter", timestamp=now)
                                alphabet_last_committed = alphabet_streak_candidate
                                alphabet_last_committed_time = now
                                alphabet_repeat_lock = True
                        else:
                            res_label = "Awaiting Sign Input"
                            res_display = "Awaiting Sign Input"
                            res_conf = 0.0
                            res_is_known = False
                            is_stable = False
                    else:
                        if alphabet_last_committed and hand_detected and has_realsign_feat:
                            res_label = alphabet_last_committed
                            res_display = f"LETTER {alphabet_last_committed}"
                            res_conf = raw_conf
                            res_is_known = True
                            is_stable = False
                        else:
                            res_label = "Awaiting Sign Input"
                            res_display = "Awaiting Sign Input"
                            res_conf = 0.0
                            res_is_known = False
                            is_stable = False

                    candidate_streak = alphabet_streak_count
                else:
                    # Word / Sequence Models (isl_mvp, isl_sequence_lstm, include_stream)
                    # If this inference produced a valid known sign, update the latched state
                    if is_pred_valid and pred_label not in ("", "no_sign", "unknown", "Awaiting Sign Input", "awaiting_sign"):
                        last_word_label = pred_label
                        if active_model == "include_stream" and include_pred is not None:
                            last_word_display = include_pred.display_name
                        elif active_model == "isl_sequence_lstm":
                            last_word_display = DISPLAY_NAMES.get(pred_label, pred_label.upper().replace("_", " "))
                        else:
                            last_word_display = DISPLAY_NAMES.get(pred_label, pred_label.upper())
                        last_word_conf = raw_conf
                        last_word_is_known = True
                        last_word_probs = res_probs
                        last_word_top3 = raw_top_3
                        last_word_pred_time = now

                    # Latch check: If hands are still visible and we have a valid word within 0.8s
                    word_latch_active = bool(
                        hand_detected
                        and last_word_is_known
                        and (now - last_word_pred_time < 0.8)
                    )

                    if word_latch_active:
                        res_label = last_word_label
                        res_display = last_word_display
                        res_conf = last_word_conf
                        res_is_known = True
                        is_stable = True
                        if not res_probs and last_word_probs:
                            res_probs = last_word_probs
                        if not raw_top_3 and last_word_top3:
                            raw_top_3 = last_word_top3

                        if candidate_label == res_label:
                            candidate_streak += 1
                        else:
                            candidate_label = res_label
                            candidate_streak = 1

                        # Disambiguation Check (when margin < 0.20 or is_ambiguous)
                        if seq_pred and getattr(seq_pred, "is_ambiguous", False) and len(seq_pred.top_3) >= 2 and not disambiguation_active:
                            t1_lbl = seq_pred.top_3[0][0]
                            t2_lbl = seq_pred.top_3[1][0]
                            if t1_lbl not in ("no_sign", "unknown") and t2_lbl not in ("no_sign", "unknown"):
                                disambiguation_active = True
                                disambiguation_top1 = t1_lbl
                                disambiguation_top2 = t2_lbl
                                disambiguation_options = [
                                    {"label": t1_lbl, "display": DISPLAY_NAMES.get(t1_lbl, t1_lbl.upper().replace("_", " ")), "hand": "left"},
                                    {"label": t2_lbl, "display": DISPLAY_NAMES.get(t2_lbl, t2_lbl.upper().replace("_", " ")), "hand": "right"},
                                ]
                                disambiguation_start_time = now

                        # Debounce gate for committing to backend accumulator (gated by disambiguation)
                        if candidate_streak >= required_streak and not disambiguation_active:
                            time_since_last = now - last_committed_time
                            is_same_sign = (res_label == last_committed_sign)
                            can_commit = False
                            if is_same_sign:
                                if hand_absent_duration >= 0.5 and had_idle_baseline:
                                    can_commit = True
                            else:
                                can_commit = True

                            if can_commit:
                                accumulator.add_token(res_display, token_type="phrase", timestamp=now)
                                last_committed_sign = res_label
                                last_committed_time = now
                                had_idle_baseline = False
                    else:
                        # Idle or Awaiting Sign
                        if not hand_detected:
                            last_word_label = ""
                            last_word_display = ""
                            last_word_conf = 0.0
                            last_word_is_known = False
                            last_word_probs = {}
                            last_word_top3 = []
                            had_idle_baseline = True

                        res_label = "Awaiting Sign Input"
                        res_display = "Signing in progress..." if is_actively_signing else ("Waiting for a hand..." if not hand_detected else "Awaiting Sign Input")
                        res_conf = 0.0
                        res_is_known = False
                        is_stable = False
                        candidate_streak = 0
                        candidate_label = ""

                # Universal Disambiguation Hand-Lift Elevation Arbiter
                if disambiguation_active:
                    resolved_choice = None
                    if has_left and len(left_hand) > 0:
                        lw_y = left_hand[0]["y"]
                        rw_y = right_hand[0]["y"] if (has_right and len(right_hand) > 0) else 1.0
                        if (lw_y < rw_y - 0.10) or (lw_y < 0.32):
                            resolved_choice = disambiguation_top1
                    if not resolved_choice and has_right and len(right_hand) > 0:
                        rw_y = right_hand[0]["y"]
                        lw_y = left_hand[0]["y"] if (has_left and len(left_hand) > 0) else 1.0
                        if (rw_y < lw_y - 0.10) or (rw_y < 0.32):
                            resolved_choice = disambiguation_top2

                    if resolved_choice:
                        is_letter = (len(resolved_choice) == 1 and resolved_choice.isalpha())
                        disp = resolved_choice.upper() if is_letter else DISPLAY_NAMES.get(resolved_choice.lower(), resolved_choice.upper().replace("_", " "))
                        token_type = "letter" if is_letter else "phrase"
                        accumulator.add_token(disp, token_type=token_type, timestamp=now)
                        disambiguation_active = False
                        disambiguation_options = []
                        disambiguation_top1 = ""
                        disambiguation_top2 = ""
                    elif now - disambiguation_start_time > 3.5:
                        disambiguation_active = False
                        disambiguation_options = []
                        disambiguation_top1 = ""
                        disambiguation_top2 = ""

                accumulator.check_timeouts(current_time=now)

                # Show-and-Sign Multimodal Sentence Fuser
                fused_sentence = None
                if (
                    pipeline.fuser is not None
                    and current_detected_objects
                    and res_label not in ("", "no_sign", "unknown", "Awaiting Sign Input", "awaiting_sign")
                ):
                    fused_sentence = pipeline.fuser.fuse(res_label, current_detected_objects)

                # Predictive Text & Phrase Suggestions
                predictive_suggestions: List[str] = []
                if pipeline.predictive is not None:
                    curr_txt = accumulator.translated_text or accumulator.sentence_buffer or ""
                    predictive_suggestions = pipeline.predictive.predict_next(curr_txt, limit=3)

                candidates_payload = []
                if raw_top_3:
                    candidates_payload = [
                        {"label": str(item[0]), "confidence": round(float(item[1]), 4)}
                        for item in raw_top_3
                        if item[0] not in ("no_sign", "no_hand", "unknown", "Awaiting Sign Input", "awaiting_sign")
                    ]

                response = {
                    "type": "prediction",
                    "active_model": active_model,
                    "label": res_label,
                    "display_name": res_display,
                    "confidence": res_conf,
                    "is_known_sign": res_is_known,
                    "is_stable": is_stable,
                    "streak_count": candidate_streak,
                    "required_streak": required_streak,
                    "probabilities": res_probs,
                    "candidates": candidates_payload,
                    "buffer_fill": buffer_fill,
                    "hand_detected": hand_detected,
                    "hands_in_subtitle_zone": hands_in_subtitle_zone,
                    "guidance": guidance_text,
                    "fps": fps,
                    "latency_ms": latency_ms,
                    "kinetic_energy": round(float(kinetic_energy), 4),
                    "is_actively_signing": bool(is_actively_signing),
                    "motion_state": "signing" if is_actively_signing else ("settled" if motion_settled_now else "idle"),
                    "sentence_buffer": accumulator.sentence_buffer,
                    "translated_text": accumulator.translated_text,
                    "sentence_tokens": list(accumulator.sentence_tokens),
                    "detected_objects": [
                        {
                            "label": o.label,
                            "display_name": o.display_name,
                            "confidence": o.confidence,
                            "is_held": o.is_held,
                            "is_interactive": o.is_interactive,
                            "bbox": o.bbox,
                        }
                        for o in current_detected_objects
                    ],
                    "fused_multimodal_text": fused_sentence,
                    "disambiguation": {
                        "active": disambiguation_active,
                        "options": disambiguation_options,
                        "top1": disambiguation_top1,
                        "top2": disambiguation_top2,
                    },
                    "lighting": {
                        "score": lighting_score,
                        "enhanced": was_low_light_enhanced,
                    },
                    "predictive_suggestions": predictive_suggestions,
                    "landmarks": {
                        "left_hand": left_hand,
                        "right_hand": right_hand,
                        "pose": pose,
                        "face": face,
                    },
                    "realsign_prediction": {
                        "letter": realsign_pred.letter if realsign_pred else "no_sign",
                        "confidence": round(realsign_pred.confidence, 4) if realsign_pred else 0.0,
                        "is_known_sign": realsign_pred.is_known_sign if realsign_pred else False,
                        "top_3": [(l, round(c, 4)) for l, c in realsign_pred.top_3] if realsign_pred else [],
                        "logits": {k: round(v, 4) for k, v in realsign_pred.logits.items()} if realsign_pred and realsign_pred.logits else None,
                    } if realsign_pred else None,
                    "sequence_lstm_prediction": {
                        "label": seq_pred.label if seq_pred else "no_sign",
                        "confidence": round(seq_pred.confidence, 4) if seq_pred else 0.0,
                        "is_known_sign": seq_pred.is_known_sign if seq_pred else False,
                        "top_3": [(l, round(c, 4)) for l, c in seq_pred.top_3] if seq_pred else [],
                    } if seq_pred else None,
                    "include_prediction": {
                        "label": include_pred.label if include_pred else "awaiting_sign",
                        "display_name": include_pred.display_name if include_pred else "Awaiting Sign...",
                        "confidence": round(include_pred.confidence, 4) if include_pred else 0.0,
                        "is_known_sign": include_pred.is_known_sign if include_pred else False,
                        "top_3": [(c.display_name, round(c.confidence, 4)) for c in include_pred.top5_candidates[:3]] if (include_pred and include_pred.top5_candidates) else [],
                        "entropy": round(include_pred.entropy, 2) if include_pred else 0.0,
                    } if include_pred else None,
                }

                await websocket.send_json(response)

            elif "text" in message and message["text"]:
                await _process_control_text(message["text"])

    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected normally.")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")
    finally:
        rolling_features.clear()
        gesture_sequence_buffer.clear()
        if streaming_include is not None:
            streaming_include.reset()
        accumulator.reset()
