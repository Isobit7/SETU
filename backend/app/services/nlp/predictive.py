"""
SETU — Predictive Text & Phrase Completion Engine
Provides statistical n-gram and prefix Trie suggestions for Indian Sign Language (ISL)
conversations, enabling 1-tap/1-gesture fast sentence completion.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("setu.services.predictive")

# Common conversational n-gram transitions in ISL domain
CONVERSATIONAL_BIGRAMS: Dict[str, List[str]] = {
    "hello": ["how are you", "good morning", "my name is", "nice to meet you"],
    "good": ["morning", "afternoon", "evening", "night", "job"],
    "good morning": ["how are you", "sir", "teacher", "have a nice day"],
    "thank": ["you", "you very much", "you so much"],
    "thank you": ["very much", "for your help", "have a good day"],
    "i": ["need water", "am hungry", "want to learn", "am fine", "do not understand", "am ready"],
    "you": ["welcome", "need help", "are kind", "understand me"],
    "water": ["please", "bottle is empty", "need to drink"],
    "help": ["me please", "needed urgently", "call doctor"],
    "namaste": ["everyone", "welcome", "how are you"],
    "what": ["is your name", "time is it", "happened", "do you want"],
    "please": ["help me", "give me water", "repeat", "wait"],
}

# Common words vocabulary for prefix completion
VOCABULARY = [
    "hello", "thank you", "good morning", "good evening", "good night",
    "namaste", "welcome", "please", "help", "water", "food", "eat",
    "drink", "hungry", "thirsty", "bathroom", "medicine", "doctor",
    "hospital", "emergency", "happy", "sad", "tired", "fine",
    "understand", "repeat", "slowly", "yes", "no", "how", "what",
    "when", "where", "who", "why", "friend", "teacher", "student",
    "name", "sign language", "india", "setu", "learn", "study",
]


class PredictiveTextEngine:
    """
    Lightweight, deterministic word and phrase prediction engine.
    Uses bigram transitions and Trie prefix matching with zero cloud latency.
    """

    def __init__(self) -> None:
        self.bigrams = CONVERSATIONAL_BIGRAMS
        self.vocab = sorted(VOCABULARY)

    def predict_next(self, current_text: str, limit: int = 3) -> List[str]:
        """
        Suggests the most probable next words or phrases based on the current sentence or last word.
        
        Args:
            current_text: Currently accumulated words or sentence.
            limit: Maximum number of suggestions to return.
            
        Returns:
            List of suggested completions.
        """
        if not current_text:
            return ["hello", "namaste", "good morning"][:limit]

        tokens = current_text.strip().lower().split()
        if not tokens:
            return ["hello", "namaste", "good morning"][:limit]

        # 1. Check bigram for last 2 tokens
        if len(tokens) >= 2:
            two_word_key = f"{tokens[-2]} {tokens[-1]}"
            if two_word_key in self.bigrams:
                return self.bigrams[two_word_key][:limit]

        # 2. Check bigram for last token
        last_token = tokens[-1]
        if last_token in self.bigrams:
            return self.bigrams[last_token][:limit]

        # 3. Check prefix matching if the user is typing/fingerspelling a partial word
        prefix_matches = [
            w for w in self.vocab if w.startswith(last_token) and w != last_token
        ]
        if prefix_matches:
            return prefix_matches[:limit]

        # 4. Fallback defaults
        return ["thank you", "please", "help"][:limit]

