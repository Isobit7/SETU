"""
SETU — Multimodal Show-and-Sign Sentence Fuser
Fuses detected physical objects (e.g. water bottle, cup, phone, book) with
real-time sign language gestures/phrases to produce coherent English output.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from backend.app.services.vision.object_detector import DetectedObject

logger = logging.getLogger("setu.services.multimodal_fuser")

# Semantic phrase mappings for common object + gesture pairings
OBJECT_SIGN_PATTERNS: Dict[Tuple[str, str], str] = {
    # Bottle
    ("bottle", "empty"): "The bottle is empty.",
    ("bottle", "water"): "I need drinking water.",
    ("bottle", "give"): "Please give me the water bottle.",
    ("bottle", "i"): "This water bottle is mine.",
    ("bottle", "you"): "This water bottle is for you.",
    ("bottle", "thank_you"): "Thank you for the water.",
    # Cup
    ("cup", "empty"): "The cup is empty.",
    ("cup", "tea"): "I would like a cup of tea.",
    ("cup", "coffee"): "I would like a cup of coffee.",
    ("cup", "give"): "Please give me the cup.",
    ("cup", "water"): "A cup of water, please.",
    # Phone
    ("phone", "call"): "Please make a call for me.",
    ("phone", "help"): "Please call emergency help.",
    ("phone", "give"): "Please hand me the phone.",
    # Book
    ("book", "open"): "Please open the book.",
    ("book", "read"): "I am reading this book.",
    ("book", "give"): "Please pass the book.",
    # Scissors
    ("scissors", "give"): "Please carefully hand me the scissors.",
    ("scissors", "cut"): "Please use scissors to cut this.",
    # Food / Fruits
    ("apple", "eat"): "I want to eat an apple.",
    ("banana", "eat"): "I want to eat a banana.",
    ("sandwich", "eat"): "I would like to have a sandwich.",
}

# Object category groups
BEVERAGE_CONTAINERS = {"bottle", "cup", "glass", "bowl"}
FOOD_ITEMS = {"apple", "banana", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake"}
ELECTRONICS = {"phone", "laptop", "mouse", "remote", "keyboard"}


class MultimodalFuser:
    """
    Fuses real-time visual object presence with gestural tokens into structured natural language.
    Handles Show-and-Sign shortcuts where showing an object replaces signing its full noun.
    """

    def __init__(self) -> None:
        pass

    def fuse(
        self,
        signed_token: str,
        detected_objects: List[DetectedObject],
        current_sentence: str = "",
    ) -> Optional[str]:
        """
        Attempts to fuse a recognized sign with the most salient interactive object.
        
        Args:
            signed_token: Normalized sign label (e.g. 'empty', 'i', 'thank_you', 'hello', 'you').
            detected_objects: List of currently detected objects from the detector.
            current_sentence: Any accumulated sentence tokens.
            
        Returns:
            Coherent fused English sentence, or None if no valid fusion rule matches.
        """
        if not signed_token or not detected_objects:
            return None

        # Pick the most prominent held interactive object
        salient_obj: Optional[DetectedObject] = None
        for obj in detected_objects:
            if obj.is_held or obj.is_interactive:
                salient_obj = obj
                break

        if salient_obj is None:
            salient_obj = detected_objects[0]

        obj_label = salient_obj.label.lower()
        sign_clean = signed_token.lower().replace(" ", "_")

        # 1. Direct dictionary match
        direct_key = (obj_label, sign_clean)
        if direct_key in OBJECT_SIGN_PATTERNS:
            return OBJECT_SIGN_PATTERNS[direct_key]

        # 2. Heuristic rules based on sign intent and object category
        if sign_clean in ("empty", "finish", "done"):
            return f"The {obj_label} is empty."

        if sign_clean in ("i", "me", "my"):
            return f"This {obj_label} belongs to me."

        if sign_clean in ("you", "your"):
            return f"This {obj_label} is for you."

        if sign_clean in ("give", "pass", "want"):
            return f"Please give me the {obj_label}."

        if sign_clean in ("thank_you", "thanks"):
            return f"Thank you for the {obj_label}."

        if sign_clean in ("good_morning", "hello", "namaste"):
            return None  # Pure greetings shouldn't combine awkwardly with random background objects

        if obj_label in FOOD_ITEMS and sign_clean in ("eat", "food", "hungry"):
            return f"I would like to eat the {obj_label}."

        if obj_label in BEVERAGE_CONTAINERS and sign_clean in ("water", "drink", "thirsty"):
            return f"I need water in the {obj_label}."

        return f"[{obj_label.capitalize()}] {signed_token.upper()}"

