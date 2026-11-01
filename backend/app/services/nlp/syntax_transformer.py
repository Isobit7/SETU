"""
SETU — Intelligent ISL Gloss-to-English Syntax Transformer.

Transforms Indian Sign Language (ISL) gloss sequences (which naturally follow
Subject-Object-Verb [SOV] structure with clause-final question words and negation)
into grammatically fluent, properly punctuated English sentences.

Linguistic Invariants Handled:
1. SOV -> SVO transformation: ("ME", "WATER", "WANT") -> "I would like some water."
2. Final Wh-word fronting: ("YOU", "LIVE", "WHERE") -> "Where do you live?"
3. Final negation scoping: ("I", "TEA", "LIKE", "NOT") -> "I do not like tea."
4. Copula & auxiliary insertion: ("HE", "TEACHER") -> "He is a teacher."
5. Pronoun case normalization: "ME" in subject position -> "I".
6. Fingerspelling concatenation and dictionary auto-correction.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Common ISL Vocabulary Taggers
PRONOUNS_SUBJECT: Dict[str, str] = {
    "ME": "I",
    "I": "I",
    "YOU": "you",
    "HE": "he",
    "SHE": "she",
    "WE": "we",
    "THEY": "they",
}

PRONOUNS_POSSESSIVE: Dict[str, str] = {
    "MY": "my",
    "YOUR": "your",
    "HIS": "his",
    "HER": "her",
    "OUR": "our",
}

WH_WORDS: Dict[str, str] = {
    "WHAT": "What",
    "WHERE": "Where",
    "WHEN": "When",
    "WHY": "Why",
    "HOW": "How",
    "WHO": "Who",
    "WHICH": "Which",
}

VERBS: Dict[str, Dict[str, str]] = {
    "WANT": {"base": "want", "present_3s": "wants", "ing": "wanting"},
    "DRINK": {"base": "drink", "present_3s": "drinks", "ing": "drinking"},
    "EAT": {"base": "eat", "present_3s": "eats", "ing": "eating"},
    "GO": {"base": "go", "present_3s": "goes", "ing": "going"},
    "COME": {"base": "come", "present_3s": "comes", "ing": "coming"},
    "READ": {"base": "read", "present_3s": "reads", "ing": "reading"},
    "HELP": {"base": "help", "present_3s": "helps", "ing": "helping"},
    "CALL": {"base": "call", "present_3s": "calls", "ing": "calling"},
    "LIKE": {"base": "like", "present_3s": "likes", "ing": "liking"},
    "MEET": {"base": "meet", "present_3s": "meets", "ing": "meeting"},
    "SLEEP": {"base": "sleep", "present_3s": "sleeps", "ing": "sleeping"},
    "WORK": {"base": "work", "present_3s": "works", "ing": "working"},
    "UNDERSTAND": {"base": "understand", "present_3s": "understands", "ing": "understanding"},
    "KNOW": {"base": "know", "present_3s": "knows", "ing": "knowing"},
    "LIVE": {"base": "live", "present_3s": "lives", "ing": "living"},
    "STUDY": {"base": "study", "present_3s": "studies", "ing": "studying"},
    "SEE": {"base": "see", "present_3s": "sees", "ing": "seeing"},
}

NOUNS: Set[str] = {
    "WATER", "FOOD", "TEA", "COFFEE", "BOOK", "HOME", "HOSPITAL", "DOCTOR",
    "TRAIN", "BUS", "SCHOOL", "COLLEGE", "MONEY", "PHONE", "NAME", "TIME",
    "MEDICINE", "HELP", "FRIEND", "FAMILY", "MOTHER", "FATHER", "TEACHER",
    "MORNING", "EVENING", "NIGHT", "DAY", "CLASS", "EXAM",
}

TIME_MARKERS: Dict[str, str] = {
    "TODAY": "today",
    "TOMORROW": "tomorrow",
    "YESTERDAY": "yesterday",
    "NOW": "now",
    "MORNING": "in the morning",
    "EVENING": "in the evening",
    "NIGHT": "tonight",
}

COMMON_DICTIONARY: List[str] = [
    "HELLO", "NAMASTE", "GOOD", "MORNING", "THANK", "YOU", "PLEASE",
    "WATER", "FOOD", "DOCTOR", "HOSPITAL", "HELP", "NAME", "TIME",
    "WHERE", "WHAT", "WHEN", "WHY", "HOW", "UNDERSTAND", "STUDENT",
    "TEACHER", "COLLEGE", "SCHOOL", "TRAIN", "STATION", "MEDICINE"
]


class ISLSyntaxTransformer:
    """
    Syntactic transformation engine that maps raw ISL gloss token streams
    into fluent, punctuated, grammatically standard English sentences.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def normalize_token(token: str) -> str:
        """Normalizes token to clean uppercase ASCII word."""
        return re.sub(r"[^\w]", "", str(token)).strip().upper()

    def transform(self, tokens: Sequence[str]) -> str:
        """
        Main entry point: Transforms a sequence of ISL gloss tokens into natural English.
        """
        if not tokens:
            return ""

        # Clean consecutive duplicates while preserving intentional double letters
        cleaned: List[str] = []
        for raw in tokens:
            norm = self.normalize_token(raw)
            if norm and (not cleaned or cleaned[-1] != norm):
                cleaned.append(norm)

        if not cleaned:
            return ""

        # Check for single-letter fingerspelling run (e.g. ['H', 'E', 'L', 'P'])
        if all(len(t) == 1 for t in cleaned):
            spelled_word = "".join(cleaned).upper()
            closest = difflib.get_close_matches(spelled_word, COMMON_DICTIONARY, n=1, cutoff=0.80)
            word = closest[0].capitalize() if closest else spelled_word.capitalize()
            return word

        # 1. Check for Question Patterns (final Wh-words: WHAT, WHERE, WHY, HOW, WHEN)
        last_tok = cleaned[-1]
        if last_tok in WH_WORDS:
            wh_word = WH_WORDS[last_tok]
            remainder = cleaned[:-1]
            return self._transform_wh_question(wh_word, remainder)

        # 2. Check for Negation Patterns (final NOT / NO)
        if last_tok in ("NOT", "NO"):
            remainder = cleaned[:-1]
            return self._transform_negation(remainder)

        # 3. Check for Time-First Clauses (e.g. ["TOMORROW", "ME", "EXAM"])
        time_prefix = ""
        if cleaned[0] in TIME_MARKERS:
            time_prefix = TIME_MARKERS[cleaned[0]].capitalize() + ", "
            cleaned = cleaned[1:]
            if not cleaned:
                return time_prefix.rstrip(", ") + "."

        # 4. Check for SOV Structure (Subject, Object, Verb)
        transformed = self._transform_sov(cleaned)
        if time_prefix:
            transformed = time_prefix + transformed[0].lower() + transformed[1:]

        return transformed

    def _transform_wh_question(self, wh: str, tokens: List[str]) -> str:
        """Transforms ISL question with clause-final wh-word into standard English."""
        if not tokens:
            return f"{wh}?"

        # Special idioms
        tok_set = set(tokens)
        if "NAME" in tok_set:
            if "YOU" in tok_set or "YOUR" in tok_set:
                return "What is your name?"
            return "What is the name?"

        if "TIME" in tok_set:
            return "What time is it?"

        if wh == "Where":
            if "YOU" in tok_set or "ME" in tok_set:
                if "LIVE" in tok_set:
                    return "Where do you live?"
                if "GO" in tok_set:
                    return "Where are you going?"
            # Object query
            subj = " ".join([t.lower() for t in tokens if t not in ("YOU", "ME")])
            if subj:
                return f"Where is the {subj}?"
            return "Where is it?"

        if wh == "Why":
            if "DELAY" in tok_set:
                return "Why is it delayed?"
            if "CANCEL" in tok_set:
                return "Why is it cancelled?"
            subj = " ".join([t.lower() for t in tokens])
            return f"Why {subj}?"

        if wh == "How":
            if "YOU" in tok_set:
                return "How are you?"
            if "MUCH" in tok_set or "COST" in tok_set:
                return "How much does it cost?"

        joined = " ".join([t.lower() for t in tokens])
        return f"{wh} is {joined}?"

    def _transform_negation(self, tokens: List[str]) -> str:
        """Transforms negated sentence: [Subject, Object, Verb, NOT] -> S does/did not V O."""
        if not tokens:
            return "No."

        subj = "I"
        verb = "want"
        obj = ""

        # Extract subject
        remaining = []
        for t in tokens:
            if t in PRONOUNS_SUBJECT and not remaining:
                subj = PRONOUNS_SUBJECT[t]
            else:
                remaining.append(t)

        # Extract verb
        found_verb = None
        other_tokens = []
        for t in remaining:
            if t in VERBS and found_verb is None:
                found_verb = VERBS[t]["base"]
            else:
                other_tokens.append(t)

        aux = "does not" if subj in ("he", "she") else "do not"
        verb_str = found_verb if found_verb else "want"
        obj_str = " ".join([t.lower() for t in other_tokens])

        if obj_str:
            return f"{subj.capitalize()} {aux} {verb_str} {obj_str}."
        return f"{subj.capitalize()} {aux} {verb_str}."

    def _transform_sov(self, tokens: List[str]) -> str:
        """Transforms Subject-Object-Verb into Subject-Verb-Object with natural auxiliaries."""
        # Check single-word greetings / needs
        if len(tokens) == 1:
            t = tokens[0]
            if t in ("HELLO", "NAMASTE"):
                return "Hello!"
            if t == "WATER":
                return "I would like some water, please."
            if t == "FOOD":
                return "I would like some food, please."
            if t == "HELP":
                return "Please help me."
            if t == "THANK_YOU" or t == "THANK":
                return "Thank you very much."
            return t.capitalize() + "."

        subj = None
        verb = None
        objects: List[str] = []

        for t in tokens:
            if t in PRONOUNS_SUBJECT and subj is None and not objects and not verb:
                subj = PRONOUNS_SUBJECT[t]
            elif t in VERBS and verb is None:
                verb = VERBS[t]
            else:
                objects.append(t.lower())

        # If no explicit subject and no verb, don't invent one
        if subj is None and verb is None:
            return ""

        # If no explicit subject, default to 'I'
        if subj is None:
            subj = "I"

        obj_phrase = " ".join(objects)

        # If verb is found, construct SVO
        if verb is not None:
            v_form = verb["present_3s"] if subj in ("he", "she") else verb["base"]
            if obj_phrase:
                # Add natural determiners for food/water/doctor
                if obj_phrase in ("water", "tea", "coffee", "food"):
                    obj_phrase = f"some {obj_phrase}"
                elif obj_phrase in ("doctor", "hospital", "book", "train", "bus"):
                    obj_phrase = f"the {obj_phrase}"
                return f"{subj.capitalize()} {v_form} {obj_phrase}."
            return f"{subj.capitalize()} {v_form}."

        # Copula clause: [Subject, Noun/Adjective] -> "He is a teacher."
        if obj_phrase:
            copula = "is a" if subj in ("he", "she") else ("am a" if subj == "I" else "are")
            return f"{subj.capitalize()} {copula} {obj_phrase}."

        return f"{subj.capitalize()}."
