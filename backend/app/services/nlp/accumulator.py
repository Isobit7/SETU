"""Real-time Sentence Accumulator & Word Timeout Buffer.

Buffers streaming letters into words, commits words after idle pauses,
and triggers ISL Gloss-to-English translation upon sentence boundaries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.app.services.nlp.gloss_parser import ISLGlossParser
from backend.app.services.nlp.neural_translator import neural_translator as default_neural_translator
from models.adapters.sequence_lstm import CTCGreedyDecoder


class SentenceAccumulator:
    """Accumulates individual alphabet letters and holistic phrases into full sentences,
    handling timing buffers, CTC continuous sequence decoding, and dispatching to neural & gloss translators.
    """

    def __init__(
        self,
        parser: Optional[ISLGlossParser] = None,
        neural_translator: Optional[Any] = None,
        word_timeout: float = 1.2,
        sentence_timeout: float = 2.5,
        ctc_blank_token: str = "no_sign",
        ctc_min_confidence: float = 0.40,
    ) -> None:
        self.parser: ISLGlossParser = parser or ISLGlossParser()
        self.neural_translator: Any = (
            neural_translator if neural_translator is not None else default_neural_translator
        )
        self.word_timeout: float = word_timeout
        self.sentence_timeout: float = sentence_timeout

        self.ctc_decoder: CTCGreedyDecoder = CTCGreedyDecoder(
            blank_token=ctc_blank_token, min_confidence=ctc_min_confidence
        )

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
        """Appends a detected sign token (letter or phrase)."""
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
            if self.current_word:
                self.sentence_tokens.append(self.current_word)
                self.current_word = ""

            self.sentence_tokens.append(clean.upper())
            self.last_token_time = timestamp
            self._tokens_changed = True

    def check_timeouts(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Checks word and sentence timeout buffers against the current time."""
        if current_time is None:
            current_time = time.time()

        word_committed = False
        sentence_translated = False

        if self.current_word and (current_time - self.last_token_time > self.word_timeout):
            self.sentence_tokens.append(self.current_word)
            self.current_word = ""
            self._tokens_changed = True
            word_committed = True

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
        """Parses the current sentence tokens through the ISL Gloss Parser and Neural Translator."""
        if not self.sentence_tokens:
            self.translated_text = ""
            return ""

        rule_match = self.parser.match_rule(self.sentence_tokens)
        if rule_match is not None:
            self.translated_text = rule_match
            return self.translated_text

        if self.neural_translator is not None and getattr(self.neural_translator, "is_loaded", False):
            self.translated_text = self.neural_translator.translate(self.sentence_tokens)
        else:
            self.translated_text = self.parser.parse_gloss(self.sentence_tokens)
        return self.translated_text

    def add_continuous_token(
        self,
        token: str,
        confidence: float = 1.0,
        token_type: str = "phrase",
        timestamp: Optional[float] = None,
    ) -> Optional[str]:
        """
        Ingests continuous streaming frame emissions using internal CTC greedy collapsing.
        Returns the committed token string if a new gesture boundary was crossed, else None.
        Enables seamless fluid co-articulation without 1.2s freeze requirements.
        """
        emitted = self.ctc_decoder.step(token, confidence=confidence)
        if emitted:
            self.add_token(emitted, token_type=token_type, timestamp=timestamp)
            return emitted
        return None

    def reset(self) -> None:
        """Clears all word, sentence, translation buffers, and CTC decoder state."""
        self.current_word = ""
        self.sentence_tokens = []
        self.last_token_time = 0.0
        self.translated_text = ""
        self._tokens_changed = False
        self.ctc_decoder.reset()

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
