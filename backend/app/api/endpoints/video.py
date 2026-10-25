"""
SETU — Video Sign Recognition API Endpoint
Handles isolated video uploads and inference via the AI4Bharat INCLUDE 263-class LSTM model.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.services.pipeline_manager import pipeline
from models.include import INCLUDEInferenceService

logger = logging.getLogger("setu.backend.api.video")

router = APIRouter(prefix="/v1/recognize", tags=["video_recognition"])

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".avi", ".webm", ".mkv"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("/video")
async def recognize_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Recognizes an isolated Indian Sign Language gesture from an uploaded video file
    using the official AI4Bharat INCLUDE 263-class pretrained model.
    """
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video extension '{ext}'. Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = Path(temp_file.name)

    try:
        bytes_read = 0
        while chunk := await file.read(1024 * 1024):
            bytes_read += len(chunk)
            if bytes_read > MAX_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded video exceeds 50MB size limit.",
                )
            temp_file.write(chunk)
        temp_file.close()

        # Ensure landmarker and include service are ready
        if not pipeline.is_ready or pipeline.landmarker is None:
            pipeline.load()
        if pipeline.landmarker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="MediaPipe Holistic vision landmarker is not available.",
            )

        service = pipeline.include_service or INCLUDEInferenceService.get_instance()
        if not service.is_loaded:
            service.load()

        result = service.predict_video(temp_path, pipeline.landmarker)
        return {
            "success": True,
            **result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during video sign recognition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition failed: {str(e)}",
        )
    finally:
        try:
            if temp_path.exists():
                os.remove(temp_path)
        except Exception as e:
            logger.warning("Could not delete temporary video file %s: %s", temp_path, e)
