"""
SETU — Dataset Recording & Custom Training Endpoints
Handles user-recorded gesture sequences (30 frames x 164D features) and dataset listing.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from training.features.extractor import FeatureExtractor158

logger = logging.getLogger("setu.backend.api.dataset")

router = APIRouter(prefix="/v1/dataset", tags=["dataset"])


class RecordSamplePayload(BaseModel):
    label: str
    sequence_164d: Optional[List[List[float]]] = None
    landmarks_sequence: Optional[List[Dict[str, Any]]] = None


@router.post("/record_sample")
async def record_sample(payload: RecordSamplePayload) -> Dict[str, Any]:
    """Accepts a 30-frame sequence of 164-D Holistic features or landmarks for custom sign collection.
    Saves as a compressed NPZ archive in data/processed/ for sequence model training.
    """
    clean_label = payload.label.strip().lower().replace(" ", "_")
    clean_label = "".join(c for c in clean_label if c.isalnum() or c in ("_", "-"))
    if not clean_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Label must not be empty.",
        )

    if payload.sequence_164d is not None:
        if len(payload.sequence_164d) != 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected sequence of exactly 30 frames, got {len(payload.sequence_164d)}.",
            )

        for i, frame in enumerate(payload.sequence_164d):
            if len(frame) != 164:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Frame {i} has dimension {len(frame)}, expected 164.",
                )

        arr = np.array(payload.sequence_164d, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sequence array contains NaN or Infinite values.",
            )
    elif payload.landmarks_sequence is not None:
        if len(payload.landmarks_sequence) != 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected sequence of exactly 30 frames, got {len(payload.landmarks_sequence)}.",
            )

        frames_164 = []
        for i, frame in enumerate(payload.landmarks_sequence):
            rh = frame.get("right_hand")
            lh = frame.get("left_hand")
            feat_158 = FeatureExtractor158.extract_from_hands(right_hand=rh, left_hand=lh)
            # Pad with 6 zeros for facial ratios to reach 164-D
            feat_164 = np.concatenate([feat_158, np.zeros(6, dtype=np.float32)])
            frames_164.append(feat_164)
        arr = np.array(frames_164, dtype=np.float32)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Expected sequence of exactly 30 frames, got none.",
        )

    save_dir = Path("data/processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"custom_{clean_label}_{timestamp}.npz"
    save_path = save_dir / filename

    np.savez_compressed(
        save_path,
        features=arr,
        label=clean_label,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    logger.info("Recorded custom sign sample '%s' saved to %s", clean_label, save_path)

    return {
        "success": True,
        "message": f"Sample for '{clean_label}' recorded successfully.",
        "label": clean_label,
        "filename": filename,
        "file_path": str(save_path),
        "frames": 30,
        "feature_dim": 164,
    }


@router.get("/custom_samples")
async def list_custom_samples() -> Dict[str, Any]:
    """Returns a list of all custom recorded sign samples in data/processed/."""
    save_dir = Path("data/processed")
    if not save_dir.exists():
        return {"samples": [], "total_samples": 0, "labels": []}

    samples = []
    labels = set()
    for p in sorted(save_dir.glob("custom_*.npz")):
        try:
            d = np.load(p)
            lbl = str(d.get("label", "unknown"))
            labels.add(lbl)
            samples.append({
                "filename": p.name,
                "label": lbl,
                "shape": list(d["features"].shape),
                "created_time": os.path.getmtime(p),
            })
        except Exception:
            continue

    return {
        "samples": samples,
        "total_samples": len(samples),
        "labels": sorted(list(labels)),
    }


class CaptureSessionPayload(BaseModel):
    label: str
    user_confirmed_label: Optional[str] = None
    session_id: Optional[str] = None
    sequence_features: Optional[List[List[float]]] = None
    lighting_score: Optional[float] = None
    latency_ms: Optional[float] = None
    fps: Optional[float] = None
    camera_angle: Optional[str] = None
    frames_base64: Optional[List[str]] = None
    user_permission_granted: bool = True


@router.post("/capture_session")
async def capture_session(payload: CaptureSessionPayload) -> Dict[str, Any]:
    """Captures an active sign translation session for dataset expansion and fine-tuning."""
    if not payload.user_permission_granted:
        return {"success": False, "message": "User permission not granted for data capture."}

    clean_label = (payload.user_confirmed_label or payload.label).strip().lower().replace(" ", "_")
    clean_label = "".join(c for c in clean_label if c.isalnum() or c in ("_", "-"))
    if not clean_label:
        clean_label = "unknown"

    today_str = datetime.date.today().isoformat()
    session_id = payload.session_id or f"session_{int(time.time())}"
    sample_id = f"{clean_label}_{int(time.time() * 1000)}"

    dest_dir = Path("data/captured_sessions") / today_str / session_id / sample_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save features if provided
    feat_dim = 164
    if payload.sequence_features and len(payload.sequence_features) > 0:
        arr = np.array(payload.sequence_features, dtype=np.float32)
        feat_dim = arr.shape[1] if arr.ndim == 2 else 164
        np.savez_compressed(
            dest_dir / "features.npz",
            features=arr,
            label=clean_label,
            lighting_score=payload.lighting_score or 0.0,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    # Save frames if provided (base64 encoded JPEG)
    saved_frames_count = 0
    if payload.frames_base64:
        import base64
        frames_dir = dest_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        for idx, b64_str in enumerate(payload.frames_base64):
            try:
                # Strip data URL prefix if present
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_str)
                frame_file = frames_dir / f"frame_{idx:03d}.jpg"
                with open(frame_file, "wb") as f_out:
                    f_out.write(raw_bytes)
                saved_frames_count += 1
            except Exception as e:
                logger.warning("Could not decode frame %d: %s", idx, e)

    # Save metadata.json
    metadata = {
        "sample_id": sample_id,
        "session_id": session_id,
        "label": payload.label,
        "user_confirmed_label": clean_label,
        "date": today_str,
        "lighting_score": payload.lighting_score,
        "latency_ms": payload.latency_ms,
        "fps": payload.fps,
        "camera_angle": payload.camera_angle or "frontal",
        "frames_saved": saved_frames_count,
        "feature_dim": feat_dim,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(dest_dir / "metadata.json", "w", encoding="utf-8") as f_meta:
        import json
        json.dump(metadata, f_meta, indent=2)

    logger.info("Captured session sample '%s' saved to %s", sample_id, dest_dir)
    return {
        "success": True,
        "sample_id": sample_id,
        "session_id": session_id,
        "directory": str(dest_dir),
        "frames_saved": saved_frames_count,
    }


@router.get("/captured_sessions")
async def list_captured_sessions() -> Dict[str, Any]:
    """Lists summary of all captured training sessions on disk."""
    base_dir = Path("data/captured_sessions")
    if not base_dir.exists():
        return {"total_sessions": 0, "total_samples": 0, "samples": []}

    samples = []
    for meta_file in sorted(base_dir.glob("**/metadata.json")):
        try:
            import json
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            samples.append({
                "sample_id": meta.get("sample_id", meta_file.parent.name),
                "label": meta.get("user_confirmed_label") or meta.get("label"),
                "date": meta.get("date"),
                "frames_saved": meta.get("frames_saved", 0),
                "lighting_score": meta.get("lighting_score"),
                "created_at": meta.get("created_at"),
            })
        except Exception:
            continue

    return {
        "total_samples": len(samples),
        "samples": samples,
    }
