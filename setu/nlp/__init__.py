"""SETU Natural Language Processing (NLP) Package.

Contains sentence accumulation, word timeout buffering, and ISL Gloss-to-English grammar parsing.
"""

from setu.nlp.accumulator import SentenceAccumulator
from setu.nlp.gloss_parser import ISLGlossParser

__all__ = ["SentenceAccumulator", "ISLGlossParser"]
