"""Unit tests for SETU NLP Pipeline.

Tests:
1. ISLGlossParser rule mapping, consecutive duplicate cleaning, and fallback formatting.
2. SentenceAccumulator real-time letter buffering, 1.2s word timeout, and 2.5s sentence translation timeout.
3. FastAPI WebSocket server integration emitting sentence_buffer and translated_text.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.services.nlp import SentenceAccumulator
from backend.app.services.nlp import ISLGlossParser
from backend.app.main import app


# ---------------------------------------------------------------------------
# 1. ISLGlossParser Tests
# ---------------------------------------------------------------------------

def test_gloss_parser_mandatory_rules():
    """Verifies all mandatory ISL SOV phrase structure mappings."""
    parser = ISLGlossParser()

    # ["NAMASTE"] -> "Hello / Greetings"
    assert parser.parse_gloss(["NAMASTE"]) == "Hello / Greetings"
    assert parser.parse_gloss(["namaste"]) == "Hello / Greetings"

    # ["ME", "WATER", "DRINK"] / ["WATER"] -> "I want water"
    assert parser.parse_gloss(["ME", "WATER", "DRINK"]) == "I want water"
    assert parser.parse_gloss(["I", "WATER", "DRINK"]) == "I want water"
    assert parser.parse_gloss(["WATER"]) == "I want water"

    # ["THANK_YOU"] -> "Thank you very much"
    assert parser.parse_gloss(["THANK_YOU"]) == "Thank you very much"
    assert parser.parse_gloss(["THANK", "YOU"]) == "Thank you very much"

    # ["HELP"] -> "Please help me"
    assert parser.parse_gloss(["HELP"]) == "Please help me"
    assert parser.parse_gloss(["help"]) == "Please help me"

    # ["NAME", "WHAT"] / ["YOUR", "NAME", "WHAT"] -> "What is your name?"
    assert parser.parse_gloss(["NAME", "WHAT"]) == "What is your name?"
    assert parser.parse_gloss(["YOUR", "NAME", "WHAT"]) == "What is your name?"
    assert parser.parse_gloss(["YOU", "NAME", "WHAT"]) == "What is your name?"


def test_gloss_parser_consecutive_duplicate_cleaning():
    """Verifies that consecutive duplicate tokens from continuous gesture holds are cleaned."""
    parser = ISLGlossParser()

    # Repeated holds: ["WATER", "WATER"] -> cleaned to ["WATER"] -> "I want water"
    assert parser.parse_gloss(["WATER", "WATER"]) == "I want water"
    assert parser.parse_gloss(["WATER", "WATER", "WATER"]) == "I want water"

    # Repeated words in phrase: ["ME", "ME", "WATER", "DRINK", "DRINK"]
    assert parser.parse_gloss(["ME", "ME", "WATER", "DRINK", "DRINK"]) == "I want water"

    # Duplicate HELP
    assert parser.parse_gloss(["HELP", "HELP", "HELP"]) == "Please help me"

    # Ensure non-consecutive duplicates are preserved if grammatically distinct
    cleaned = parser.clean_duplicate_tokens(["HELLO", "ME", "HELLO"])
    assert cleaned == ["HELLO", "ME", "HELLO"]


def test_gloss_parser_fallback_capitalized_sentence():
    """Verifies fallback to clean capitalized sentence string when no rule matches."""
    parser = ISLGlossParser()

    # Unmapped tokens
    res = parser.parse_gloss(["UNMAPPED", "GESTURE", "TOKEN"])
    assert res == "Unmapped gesture token"

    # Lowercase / mixed tokens
    res2 = parser.parse_gloss(["custom", "sign"])
    assert res2 == "Custom sign"

    # Empty tokens
    assert parser.parse_gloss([]) == ""
    assert parser.parse_gloss(["", "  "]) == ""


def test_gloss_parser_custom_rules():
    """Verifies that custom rule mappings can be supplied to the parser."""
    custom_rules = {
        ("TEACHER", "COME"): "The teacher has arrived.",
    }
    parser = ISLGlossParser(rules=custom_rules)
    assert parser.parse_gloss(["TEACHER", "COME"]) == "The teacher has arrived."
    # Standard rules still work
    assert parser.parse_gloss(["NAMASTE"]) == "Hello / Greetings"


# ---------------------------------------------------------------------------
# 2. SentenceAccumulator Tests
# ---------------------------------------------------------------------------

def test_accumulator_letter_buffering():
    """Verifies that letter tokens buffer into current_word without prematurely committing."""
    acc = SentenceAccumulator()

    acc.add_token("H", token_type="letter", timestamp=10.0)
    acc.add_token("e", token_type="letter", timestamp=10.3)
    acc.add_token("l", token_type="letter", timestamp=10.6)
    acc.add_token("p", token_type="letter", timestamp=10.9)

    assert acc.current_word == "HELP"
    assert acc.sentence_tokens == []
    assert acc.sentence_buffer == "HELP"
    assert acc.last_token_time == 10.9


def test_accumulator_word_timeout_commit():
    """Verifies word timeout buffer: commits current_word to sentence_tokens after 1.2s idle."""
    acc = SentenceAccumulator(word_timeout=1.2, sentence_timeout=2.5)

    # Spell "WATER" at t=10.0 to 10.8
    acc.add_token("W", "letter", timestamp=10.0)
    acc.add_token("A", "letter", timestamp=10.2)
    acc.add_token("T", "letter", timestamp=10.4)
    acc.add_token("E", "letter", timestamp=10.6)
    acc.add_token("R", "letter", timestamp=10.8)

    # At t=11.5 (dt = 0.7s < 1.2s): no word timeout
    status_1 = acc.check_timeouts(current_time=11.5)
    assert status_1["word_committed"] is False
    assert acc.current_word == "WATER"
    assert acc.sentence_tokens == []

    # At t=12.1 (dt = 1.3s > 1.2s): word timeout fires!
    status_2 = acc.check_timeouts(current_time=12.1)
    assert status_2["word_committed"] is True
    assert acc.current_word == ""
    assert acc.sentence_tokens == ["WATER"]
    assert acc.sentence_buffer == "WATER"
    # Sentence translation not triggered yet (12.1 - 10.8 = 1.3s < 2.5s)
    assert status_2["sentence_translated"] is False
    assert acc.translated_text == ""


def test_accumulator_sentence_timeout_triggers_translation():
    """Verifies sentence timeout buffer: triggers translation after 2.5s idle."""
    acc = SentenceAccumulator(word_timeout=1.2, sentence_timeout=2.5)

    # User signs phrase "NAMASTE" at t=100.0
    acc.add_token("NAMASTE", "phrase", timestamp=100.0)
    assert acc.sentence_tokens == ["NAMASTE"]

    # At t=102.0 (dt = 2.0s < 2.5s): sentence timeout not reached
    s1 = acc.check_timeouts(current_time=102.0)
    assert s1["sentence_translated"] is False
    assert acc.translated_text == ""

    # At t=102.6 (dt = 2.6s > 2.5s): sentence timeout triggers translation!
    s2 = acc.check_timeouts(current_time=102.6)
    assert s2["sentence_translated"] is True
    assert acc.translated_text == "Hello / Greetings"
    assert s2["translated_text"] == "Hello / Greetings"


def test_accumulator_full_letter_to_translated_sentence_pipeline():
    """End-to-end pipeline: fingerspells letters -> word timeout -> sentence timeout -> translated English."""
    acc = SentenceAccumulator(word_timeout=1.2, sentence_timeout=2.5)

    # 1. Spell "H", "E", "L", "P" at t=0.0
    for ch in "HELP":
        acc.add_token(ch, "letter", timestamp=0.0)

    # 2. Advance time past 1.2s to commit word
    acc.check_timeouts(current_time=1.3)
    assert acc.sentence_tokens == ["HELP"]
    assert acc.current_word == ""

    # 3. Advance time past 2.5s to trigger sentence translation
    res = acc.check_timeouts(current_time=2.6)
    assert res["sentence_translated"] is True
    assert acc.translated_text == "Please help me"


def test_accumulator_phrase_commits_pending_word():
    """Verifies that adding a phrase token immediately commits any pending fingerspelled word."""
    acc = SentenceAccumulator(word_timeout=1.2, sentence_timeout=2.5)

    # Finger-spell "ME"
    acc.add_token("M", "letter", timestamp=10.0)
    acc.add_token("E", "letter", timestamp=10.2)
    assert acc.current_word == "ME"

    # Now perform phrase "WATER"
    acc.add_token("WATER", "phrase", timestamp=10.5)
    assert acc.current_word == ""
    assert acc.sentence_tokens == ["ME", "WATER"]

    # Now perform phrase "DRINK"
    acc.add_token("DRINK", "phrase", timestamp=11.0)
    assert acc.sentence_tokens == ["ME", "WATER", "DRINK"]

    # Timeout after 2.5s
    acc.check_timeouts(current_time=13.6)
    assert acc.translated_text == "I want water"


def test_accumulator_reset_and_backspace():
    """Verifies reset() and backspace() operations."""
    acc = SentenceAccumulator()

    acc.add_token("A", "letter")
    acc.add_token("B", "letter")
    assert acc.current_word == "AB"

    acc.backspace()
    assert acc.current_word == "A"

    acc.backspace()
    assert acc.current_word == ""

    # Test backspace on sentence_tokens
    acc.add_token("HELLO", "phrase")
    acc.add_token("WORLD", "phrase")
    assert acc.sentence_tokens == ["HELLO", "WORLD"]

    acc.backspace()
    assert acc.sentence_tokens == ["HELLO"]

    acc.reset()
    assert acc.current_word == ""
    assert acc.sentence_tokens == []
    assert acc.translated_text == ""
    assert acc.sentence_buffer == ""


# ---------------------------------------------------------------------------
# 3. Stream Server WebSocket Integration Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as c:
        yield c


def test_server_websocket_emits_sentence_buffer_and_translated_text(api_client):
    """Verifies that the WebSocket endpoint /api/v1/ws/recognize emits sentence_buffer and translated_text."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", blank)
    jpeg_bytes = encoded.tobytes()

    with api_client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Send a mock video frame
        ws.send_bytes(jpeg_bytes)
        resp = ws.receive_json()

        assert resp["type"] == "prediction"
        assert "sentence_buffer" in resp
        assert "translated_text" in resp
        assert "sentence_tokens" in resp
        assert isinstance(resp["sentence_buffer"], str)
        assert isinstance(resp["translated_text"], str)

        # Test clear_sentence action
        ws.send_text('{"action": "clear_sentence"}')
        info_resp = ws.receive_json()
        assert info_resp["type"] == "info"
        assert "Sentence buffer cleared" in info_resp["message"]
        assert info_resp["sentence_buffer"] == ""

        # Test backspace action
        ws.send_text('{"action": "backspace"}')
        backspace_resp = ws.receive_json()
        assert backspace_resp["type"] == "info"
        assert "Backspace applied" in backspace_resp["message"]


def test_accumulator_continuous_ctc_fluid_signing():
    """Verifies that continuous streaming frame emissions decode into clean sentences via CTC without manual pauses."""
    acc = SentenceAccumulator(ctc_blank_token="no_sign", ctc_min_confidence=0.40)

    # Simulated continuous frame emissions:
    # 3 frames idle resting -> 4 frames HELLO -> 2 frames inter-sign transition -> 4 frames YOU -> 2 frames idle
    frame_stream = [
        ("no_sign", 0.95),
        ("no_sign", 0.90),
        ("no_sign", 0.92),
        ("HELLO", 0.88),
        ("HELLO", 0.91),
        ("HELLO", 0.89),
        ("HELLO", 0.93),
        ("no_sign", 0.60),  # transitional movement
        ("no_sign", 0.75),
        ("YOU", 0.85),
        ("YOU", 0.89),
        ("YOU", 0.87),
        ("no_sign", 0.95),
    ]

    emitted_tokens = []
    for tok, conf in frame_stream:
        committed = acc.add_continuous_token(tok, confidence=conf, token_type="phrase")
        if committed:
            emitted_tokens.append(committed)

    # Exactly 2 tokens should be emitted despite 13 total frames
    assert emitted_tokens == ["hello", "you"]
    assert acc.sentence_buffer == "HELLO YOU"

    # Translation runs smoothly
    translated = acc.translate_sentence()
    assert "Hello" in translated or "Greetings" in translated


def test_continuous_multi_sign_transitions_without_manual_pauses():
    """Verification Gate 4: Continuous co-articulated signing transitions emit clean word boundaries without freeze holds."""
    acc = SentenceAccumulator()

    continuous_gestures = [
        ("THANK_YOU", 0.90),
        ("THANK_YOU", 0.92),
        ("no_sign", 0.80),
        ("I", 0.85),
        ("I", 0.88),
        ("no_sign", 0.70),
        ("HELP", 0.89),
        ("HELP", 0.92),
    ]

    for gesture, conf in continuous_gestures:
        acc.add_continuous_token(gesture, confidence=conf, token_type="phrase")

    assert acc.sentence_tokens == ["THANK_YOU", "I", "HELP"]
    assert acc.sentence_buffer == "THANK_YOU I HELP"

