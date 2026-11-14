"""ISL Gloss-to-English Grammar Parser.

Maps Indian Sign Language (ISL) gloss tokens (typically Subject-Object-Verb [SOV]
or isolated signs) to fluent English (Subject-Verb-Object [SVO]) grammatical sentences.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_ISL_RULES: Dict[Tuple[str, ...], str] = {
    # Greetings & Courtesy
    ("NAMASTE",): "Hello / Greetings",
    ("HELLO",): "Hello / Greetings",
    ("GOOD_MORNING",): "Good morning",
    ("GOOD", "MORNING"): "Good morning",
    ("GOOD_AFTERNOON",): "Good afternoon",
    ("GOOD", "AFTERNOON"): "Good afternoon",
    ("GOOD_NIGHT",): "Good night",
    ("GOOD", "NIGHT"): "Good night",
    ("THANK_YOU",): "Thank you very much",
    ("THANK", "YOU"): "Thank you very much",
    ("PLEASE",): "Please",
    ("SORRY",): "I am sorry",
    ("WELCOME",): "You are welcome",
    ("YES",): "Yes",
    ("NO",): "No",
    ("HELP",): "Please help me",
    ("PLEASE", "HELP"): "Please help me",
    ("PLEASE", "HELP", "ME"): "Please help me",
    ("HELP", "ME"): "Please help me",

    # Needs & Actions (ISL SOV -> English SVO)
    ("ME", "WATER", "DRINK"): "I want water",
    ("I", "WATER", "DRINK"): "I want water",
    ("WATER", "DRINK"): "I want water",
    ("WATER", "WANT"): "I want water",
    ("ME", "WATER", "WANT"): "I want water",
    ("I", "WATER", "WANT"): "I want water",
    ("WATER",): "I want water",
    ("ME", "FOOD", "EAT"): "I want food",
    ("I", "FOOD", "EAT"): "I want food",
    ("FOOD", "EAT"): "I want food",
    ("FOOD", "WANT"): "I want food",
    ("ME", "FOOD", "WANT"): "I want food",
    ("I", "FOOD", "WANT"): "I want food",
    ("FOOD",): "I want food",
    ("EAT",): "I want to eat",
    ("DRINK",): "I want to drink",
    ("I", "HELP", "NEED"): "I need help",
    ("ME", "HELP", "NEED"): "I need help",
    ("HELP", "NEED"): "I need help",
    ("DOCTOR", "NEED"): "I need a doctor",
    ("MEDICINE", "NEED"): "I need medicine",
    ("DOCTOR",): "I need a doctor",
    ("MEDICINE",): "I need medicine",

    # Questions & Inquiries
    ("NAME", "WHAT"): "What is your name?",
    ("YOUR", "NAME", "WHAT"): "What is your name?",
    ("YOU", "NAME", "WHAT"): "What is your name?",
    ("HOW", "ARE", "YOU"): "How are you?",
    ("HOW", "YOU"): "How are you?",
    ("YOU", "HOW"): "How are you?",
    ("WHERE", "YOU", "GO"): "Where are you going?",
    ("WHERE", "GO"): "Where are you going?",
    ("YOU", "LIVE", "WHERE"): "Where do you live?",
    ("YOU", "COME", "WHEN"): "When will you come?",
    ("DOCTOR", "WHERE"): "Where is the doctor?",
    ("HOSPITAL", "WHERE"): "Where is the hospital?",
    ("BATHROOM", "WHERE"): "Where is the restroom?",
    ("TOILET", "WHERE"): "Where is the restroom?",
    ("TIME", "WHAT"): "What time is it?",

    # Common Expressions & Feelings
    ("NICE", "MEET", "YOU"): "Nice to meet you",
    ("SEE", "YOU", "LATER"): "See you later",
    ("I", "LOVE", "YOU"): "I love you",
    ("I", "ME"): "I / Me",
    ("I", "HAPPY"): "I am happy",
    ("ME", "HAPPY"): "I am happy",
    ("I", "SAD"): "I am sad",
    ("ME", "SAD"): "I am sad",
    ("I", "FINE"): "I am fine",
    ("ME", "FINE"): "I am fine",
    ("GOOD",): "Good",
    ("BAD",): "Bad",
    ("I", "TEA", "LIKE", "NOT"): "I do not like tea",
}


class ISLGlossParser:
    """Parses ISL gloss token sequences into fluent English grammar."""

    def __init__(self, rules: Optional[Dict[Tuple[str, ...], str]] = None) -> None:
        """Initializes parser with standard or custom ISL translation rules."""
        self.rules: Dict[Tuple[str, ...], str] = dict(DEFAULT_ISL_RULES)
        if rules:
            for pattern, translation in rules.items():
                normalized_key = tuple(self._normalize_token(t) for t in pattern if self._normalize_token(t))
                if normalized_key:
                    self.rules[normalized_key] = translation

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalizes a gloss token by trimming, uppercasing, and standardizing separators."""
        if not token:
            return ""
        cleaned = re.sub(r"[^\w\s-]", "", str(token)).strip().upper()
        cleaned = re.sub(r"[\s-]+", "_", cleaned)
        return cleaned

    def clean_duplicate_tokens(self, tokens: Sequence[str]) -> List[str]:
        """Cleans duplicate consecutive tokens from the input sequence."""
        cleaned: List[str] = []
        for raw in tokens:
            normalized = self._normalize_token(raw)
            if not normalized:
                continue
            if not cleaned or cleaned[-1] != normalized:
                cleaned.append(normalized)
        return cleaned

    def match_rule(self, tokens: Sequence[str]) -> Optional[str]:
        """Returns canonical translation if tokens match an exact dictionary rule, else None."""
        if not tokens:
            return None

        cleaned = self.clean_duplicate_tokens(tokens)
        if not cleaned:
            return None

        key = tuple(cleaned)
        if key in self.rules:
            return self.rules[key]

        single_token_key = ("_".join(cleaned),)
        if single_token_key in self.rules:
            return self.rules[single_token_key]

        expanded_tokens: List[str] = []
        for t in cleaned:
            parts = t.split("_")
            expanded_tokens.extend([p for p in parts if p])
        expanded_key = tuple(expanded_tokens)
        if expanded_key in self.rules:
            return self.rules[expanded_key]

        return None

    def parse_gloss(self, tokens: Sequence[str]) -> str:
        """Parses a list of ISL gloss tokens into English."""
        rule_match = self.match_rule(tokens)
        if rule_match is not None:
            return rule_match

        cleaned = self.clean_duplicate_tokens(tokens)
        try:
            from backend.app.services.nlp.syntax_transformer import ISLSyntaxTransformer
            transformed = ISLSyntaxTransformer().transform(cleaned)
            if transformed:
                return transformed
        except Exception:
            pass
        return self._fallback_sentence(cleaned)

    def _fallback_sentence(self, cleaned: List[str]) -> str:
        """Formats unmapped gloss tokens as a clean capitalized English sentence."""
        words: List[str] = []
        for token in cleaned:
            parts = token.split("_")
            for p in parts:
                if p:
                    words.append(p.lower())

        if not words:
            return ""

        raw_sentence = " ".join(words)
        return raw_sentence[0].upper() + raw_sentence[1:]
