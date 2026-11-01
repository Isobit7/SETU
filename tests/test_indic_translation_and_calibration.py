"""
SETU — Multilingual Indic Language Translation & Signer Calibration Test Suite
Validates:
1. PROB-17: Accurate SOV grammar transformation into Hindi, Tamil, Telugu, and Kannada.
2. PROB-17: Idioms, questions, negations, and single-word token gloss translation.
3. PROB-17: SentenceAccumulator multilingual translations payload emission.
4. PROB-19: Signer Personal Calibration Wizard (rest_y thresholding and WebSocket control).
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.nlp.indic_transformer import indic_transformer
from backend.app.services.nlp.accumulator import SentenceAccumulator
from training.features.geometry import is_resting_or_neutral_pose


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. PROB-17: INDIC TRANSFORMER CANONICAL IDIOMS & PHRASES
# ---------------------------------------------------------------------------

def test_indic_transformer_canonical_idioms():
    """Validates exact translation of foundational ISL greetings and social signs."""
    # Hindi greetings
    assert indic_transformer.translate(["HELLO"], lang="hi") == "नमस्ते!"
    assert indic_transformer.translate(["NAMASTE"], lang="hi") == "नमस्ते।"
    assert indic_transformer.translate(["THANK", "YOU"], lang="hi") == "धन्यवाद।"
    assert indic_transformer.translate(["PLEASE"], lang="hi") == "कृपया।"
    assert indic_transformer.translate(["SORRY"], lang="hi") == "माफ़ कीजिए।"

    # Tamil greetings
    assert indic_transformer.translate(["NAMASTE"], lang="ta") == "வணக்கம்."
    assert indic_transformer.translate(["THANK", "YOU"], lang="ta") == "நன்றி."

    # Telugu greetings
    assert indic_transformer.translate(["NAMASTE"], lang="te") == "నమస్కారం."
    assert indic_transformer.translate(["THANK", "YOU"], lang="te") == "ధన్యవాదాలు."

    # Kannada greetings
    assert indic_transformer.translate(["NAMASTE"], lang="kn") == "ನಮಸ್ಕಾರ."
    assert indic_transformer.translate(["THANK", "YOU"], lang="kn") == "ಧನ್ಯವಾದಗಳು."


def test_indic_transformer_questions_and_sov_composition():
    """Validates SOV grammatical structure and question particle generation."""
    # Questions
    q_name = indic_transformer.translate(["YOU", "NAME", "WHAT"], lang="hi")
    assert q_name == "आपका नाम क्या है?"

    q_live = indic_transformer.translate(["YOU", "LIVE", "WHERE"], lang="hi")
    assert q_live == "आप कहाँ रहते हैं?"

    q_doctor = indic_transformer.translate(["DOCTOR", "WHERE"], lang="hi")
    assert q_doctor == "डॉक्टर कहाँ हैं?"

    # Daily needs / desires
    s_water = indic_transformer.translate(["ME", "WATER", "WANT"], lang="hi")
    assert s_water == "मुझे पानी चाहिए।"

    s_help = indic_transformer.translate(["I", "HELP", "NEED"], lang="hi")
    assert s_help == "मुझे मदद चाहिए।"

    # Negation
    s_tea = indic_transformer.translate(["I", "TEA", "LIKE", "NOT"], lang="hi")
    assert s_tea == "मुझे चाय पसंद नहीं है।"


def test_indic_transformer_translate_all_languages():
    """translate_all returns valid outputs for all 4 supported Indic languages."""
    tokens = ["THANK", "YOU"]
    translations = indic_transformer.translate_all(tokens)

    assert "hi" in translations and translations["hi"] == "धन्यवाद।"
    assert "ta" in translations and translations["ta"] == "நன்றி."
    assert "te" in translations and translations["te"] == "ధన్యవాదాలు."
    assert "kn" in translations and translations["kn"] == "ಧನ್ಯವಾದಗಳು."


# ---------------------------------------------------------------------------
# 2. PROB-17: SENTENCE ACCUMULATOR INDIC INTEGRATION
# ---------------------------------------------------------------------------

def test_sentence_accumulator_multilingual_emission():
    """SentenceAccumulator automatically generates and caches Indic translations."""
    acc = SentenceAccumulator(word_timeout=0.01, sentence_timeout=0.01)

    acc.add_token("NAMASTE", token_type="phrase", timestamp=100.0)
    acc.check_timeouts(current_time=105.0)

    assert "hi" in acc.translated_indic
    assert acc.translated_indic["hi"] == "नमस्ते।"
    assert acc.translated_indic["ta"] == "வணக்கம்."

    # Test reset clears Indic translations
    acc.reset()
    assert len(acc.translated_indic) == 0


# ---------------------------------------------------------------------------
# 3. PROB-17 & PROB-19: WEBSOCKET CONTROL & CALIBRATION
# ---------------------------------------------------------------------------

def test_ws_set_language_control_command(client):
    """User selects target translation language via WebSocket control action."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Set to Hindi
        ws.send_text(json.dumps({"action": "set_language", "language": "hi"}))
        r_hi = ws.receive_json()
        assert r_hi["type"] == "info"
        assert r_hi["target_language"] == "hi"

        # Set to Tamil
        ws.send_text(json.dumps({"action": "set_language", "language": "ta-IN"}))
        r_ta = ws.receive_json()
        assert r_ta["type"] == "info"
        assert r_ta["target_language"] == "ta-in"


def test_ws_personal_calibration_wizard(client):
    """User runs 5-second calibration wizard sending custom rest_y threshold."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        # Valid calibration for a low desk / tall user
        ws.send_text(json.dumps({"action": "set_calibration", "rest_y": 0.74}))
        r_cal = ws.receive_json()
        assert r_cal["type"] == "info"
        assert r_cal["rest_y"] == 0.74
        assert "Personal calibration updated" in r_cal["message"]

        # Out-of-bounds calibration clamped to [0.50, 0.95]
        ws.send_text(json.dumps({"action": "set_calibration", "rest_y": 0.20}))
        r_low = ws.receive_json()
        assert r_low["rest_y"] == 0.50

        ws.send_text(json.dumps({"action": "set_calibration", "rest_y": 1.45}))
        r_high = ws.receive_json()
        assert r_high["rest_y"] == 0.95


def test_custom_rest_y_threshold_behavior():
    """Demonstrates that calibrated rest_y allows custom desk heights to be rejected as rest."""
    # A hand at y = 0.76 (e.g. high desk or low camera)
    hand_at_76 = [{"x": 0.5, "y": 0.76 + i * 0.002, "z": 0.0} for i in range(21)]

    # With default threshold 0.82, hand at 0.76 is NOT rejected (is considered active)
    assert is_resting_or_neutral_pose(hand_at_76, y_threshold=0.82) is False

    # With calibrated threshold 0.74, hand at 0.76 IS properly recognized as resting!
    assert is_resting_or_neutral_pose(hand_at_76, y_threshold=0.74) is True
