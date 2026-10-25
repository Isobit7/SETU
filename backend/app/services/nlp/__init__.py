"""
SETU — Natural Language Processing & Gloss Translation Services
"""

from backend.app.services.nlp.gloss_parser import ISLGlossParser
from backend.app.services.nlp.accumulator import SentenceAccumulator

__all__ = [
    "ISLGlossParser",
    "SentenceAccumulator",
]
