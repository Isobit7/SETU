"""Real-time Sentence Accumulator & Word Timeout Buffer.

Buffers streaming letters into words, commits words after idle pauses,
and triggers ISL Gloss-to-English translation upon sentence boundaries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from setu.nlp.gloss_parser import ISLGlossParser


class SentenceAccumulator:
    """Accumulates individual alphabet letters and holistic phrases into full sentences,

    handling timing buffers and dispatching to the ISLGlossParser.
    """

    def __init__(
        self,
        parser: Optional[ISLGlossParser] = None,
        word_timeout: float = 1.2,
        sentence_timeout: float = 2.5,
    ) -> None:
        """Initializes accumulator with specified timeouts.

        Args:
            parser: ISLGlossParser instance for translating accumulated gloss tokens.
            word_timeout: Idle duration in seconds after which letters commit into a word (default: 1.2s).
            sentence_timeout: Idle duration in seconds after which a sentence translates (default: 2.5s).
        """
        self.parser: ISLGlossParser = parser or ISLGlossParser()
        self.word_timeout: float = word_timeout
        self.sentence_timeout: float = sentence_timeout

        self.current_word: str = ""
        self.sentence_tokens: List[str] = []
        self.last_token_time: float = 0.0
        self.translated_text: str = ""
        self._tokens_changed: bool = False

    @property
    def sentence_buffer(self) -> str:
        """Returns the current accumulated sentence buffer as a string."""
        tokens = list(self.sentence_tokens)
        if self.current_word:
            tokens.append(self.current_word)
        return " ".join(tokens)

    def add_token(
        self,
        token: str,
        token_type: str = "letter",
        timestamp: Optional[float] = None,
    ) -> None:
        """Appends a detected sign token (letter or phrase).

        Args:
            token: Detected character (e.g. 'H') or phrase (e.g. 'NAMASTE').
            token_type: Either 'letter' or 'phrase'.
            timestamp: Event epoch timestamp in seconds. Defaults to time.time().
        """
        if timestamp is None:
            timestamp = time.time()

        clean = str(token).strip()
        if not clean:
            return

        normalized_type = str(token_type).lower().strip()

        if normalized_type == "letter":
            self.current_word += clean.upper()
            self.last_token_time = timestamp
            self._tokens_changed = True
        elif normalized_type == "phrase":
            # If a word was being finger-spelled, commit it first
            if self.current_word:
                self.sentence_tokens.append(self.current_word)
                self.current_word = ""

            self.sentence_tokens.append(clean.upper())
            self.last_token_time = timestamp
            self._tokens_changed = True

    def check_timeouts(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Checks word and sentence timeout buffers against the current time.

        - If current_word exists and current_time - last_token_time > 1.2s:
            Commit current_word to sentence_tokens.
        - If sentence_tokens changed and current_time - last_token_time > 2.5s:
            Trigger translate_sentence().

        Args:
            current_time: Current epoch time in seconds. Defaults to time.time().

        Returns:
            Dict containing timeout flags and latest state.
        """
        if current_time is None:
            current_time = time.time()

        word_committed = False
        sentence_translated = False

        # 1. Word timeout check: commit pending finger-spelled letters
        if self.current_word and (current_time - self.last_token_time > self.word_timeout):
            self.sentence_tokens.append(self.current_word)
            self.current_word = ""
            self._tokens_changed = True
            word_committed = True

        # 2. Sentence timeout check: trigger ISL-to-English grammar translation
        if self._tokens_changed and (current_time - self.last_token_time > self.sentence_timeout):
            if self.sentence_tokens:
                self.translate_sentence()
                sentence_translated = True
            self._tokens_changed = False

        return {
            "word_committed": word_committed,
            "sentence_translated": sentence_translated,
            "sentence_buffer": self.sentence_buffer,
            "translated_text": self.translated_text,
        }

    def translate_sentence(self) -> str:
        """Parses the current sentence tokens through the ISL Gloss Parser."""
        if not self.sentence_tokens:
            self.translated_text = ""
            return ""

        self.translated_text = self.parser.parse_gloss(self.sentence_tokens)
        return self.translated_text

    def reset(self) -> None:
        """Clears all word, sentence, and translation buffers."""
        self.current_word = ""
        self.sentence_tokens = []
        self.last_token_time = 0.0
        self.translated_text = ""
        self._tokens_changed = False

    def clear(self) -> None:
        """Alias for reset()."""
        self.reset()

    def backspace(self) -> None:
        """Removes the last letter from current_word, or the last token from sentence_tokens."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.last_token_time = time.time()
            self._tokens_changed = True
        elif self.sentence_tokens:
            self.sentence_tokens.pop()
            self.last_token_time = time.time()
            self._tokens_changed = True
