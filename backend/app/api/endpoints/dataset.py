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
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger("setu.backend.api.dataset")

router = APIRouter(prefix="/v1/dataset", tags=["dataset"])


class RecordSamplePayload(BaseModel):
    label: str
    sequence_164d: List[List[float]]


@router.post("/record_sample")
async def record_sample(payload: RecordSamplePayload) -> Dict[str, Any]:
    """Accepts a 30-frame sequence of 164-D Holistic + Facial features for custom sign collection.
    Saves as a compressed NPZ archive in data/processed/ for sequence model training.
    """
    clean_label = payload.label.strip().lower().replace(" ", "_")
    clean_label = "".join(c for c in clean_label if c.isalnum() or c in ("_", "-"))
    if not clean_label:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Label must not be empty.",
        )

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
