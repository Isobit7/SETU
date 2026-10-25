"""
SETU — Vocabulary API Endpoint
Exposes supported signs, categories, display titles, and gesture descriptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from backend.app.services.pipeline_manager import pipeline

logger = logging.getLogger("setu.backend.api.vocabulary")

router = APIRouter(prefix="/v1", tags=["vocabulary"])

VOCABULARY_METADATA = [
    {"key": "hello", "display": "HELLO", "category": "Greeting", "description": "Open hand greeting wave or salute"},
    {"key": "thank_you", "display": "THANK YOU", "category": "Courtesy", "description": "Flat hand moving forward from chin/lips"},
    {"key": "good_morning", "display": "GOOD MORNING", "category": "Greeting", "description": "Combined gesture for good morning"},
    {"key": "i", "display": "I / ME", "category": "Pronoun", "description": "Index finger pointing to chest / me"},
    {"key": "you", "display": "YOU", "category": "Pronoun", "description": "Index finger pointing forward to conversation partner"},
]


@router.get("/vocabulary")
async def get_vocabulary() -> Dict[str, Any]:
    """Returns supported signs with display titles and description."""
    if not pipeline.classifier:
        return {"vocabulary": []}
    return {
        "vocabulary": VOCABULARY_METADATA,
        "negative_class": "no_sign",
        "total_classes": len(pipeline.classifier.class_names),
    }
