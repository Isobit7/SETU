"""
SETU — Digits 0–9, Currency Formatting & Domain Vocabulary Packs Test Suite
Validates:
1. PROB-18: ISL Digit Classifier (anatomical joint flexion & extension for 0-9).
2. PROB-18: Consecutive digit accumulation and Indian Rupee currency prefix formatting.
3. PROB-24: Healthcare & Clinical Domain Vocabulary Pack (symptoms, personnel, facilities).
4. PROB-24: Banking & Financial Domain Vocabulary Pack (transactions, accounts, identity).
5. PROB-24: WebSocket domain switching and context-aware predictive suggestions.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from models.adapters.digits import digit_classifier, DigitPrediction
from backend.app.services.nlp.domain_packs import (
    format_tokens_with_numbers_and_currency,
    domain_packs_manager,
    HEALTHCARE_PACK,
    BANKING_PACK,
)
from backend.app.services.nlp.accumulator import SentenceAccumulator


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. PROB-18: ANATOMICAL ISL DIGIT CLASSIFIER
# ---------------------------------------------------------------------------

def test_isl_digit_classifier_extensions():
    """Validates anatomical joint angle discrimination for single-hand digits."""
    def make_digit_hand(extended_indices: list[int]) -> np.ndarray:
        hand = np.zeros((21, 3), dtype=np.float32)
        finger_chains = {
            1: [1, 2, 3, 4],       # Thumb
            2: [5, 6, 7, 8],       # Index
            3: [9, 10, 11, 12],    # Middle
            4: [13, 14, 15, 16],   # Ring
            5: [17, 18, 19, 20],   # Pinky
        }
        for f_num, chain in finger_chains.items():
            if f_num in extended_indices:
                # Extended phalanx
                hand[chain[0]] = [0.0, -0.4, 0.0]
                hand[chain[1]] = [0.0, -0.6, 0.0]
                hand[chain[2]] = [0.0, -0.8, 0.0]
                hand[chain[3]] = [0.0, -1.0, 0.0]
            else:
                # Tightly curled phalanx
                hand[chain[0]] = [0.0, -0.4, 0.0]
                hand[chain[1]] = [0.0, -0.6, 0.0]
                hand[chain[2]] = [0.0, -0.5, 0.1]
                hand[chain[3]] = [0.0, -0.4, 0.1]
        return hand

    # Digit 1: Only Index extended (finger 2)
    pred1 = digit_classifier.predict_landmarks(make_digit_hand([2]))
    assert pred1.digit == "1"
    assert pred1.confidence >= 0.85

    # Digit 2: Index + Middle extended (fingers 2, 3)
    pred2 = digit_classifier.predict_landmarks(make_digit_hand([2, 3]))
    assert pred2.digit == "2"
    assert pred2.confidence >= 0.85

    # Digit 4: Four fingers extended without thumb (fingers 2, 3, 4, 5)
    pred4 = digit_classifier.predict_landmarks(make_digit_hand([2, 3, 4, 5]))
    assert pred4.digit == "4"
    assert pred4.confidence >= 0.85

    # Digit 5: All 5 fingers extended
    pred5 = digit_classifier.predict_landmarks(make_digit_hand([1, 2, 3, 4, 5]))
    assert pred5.digit == "5"
    assert pred5.confidence >= 0.85

    # Empty/degenerate hand
    empty_pred = digit_classifier.predict_landmarks([])
    assert empty_pred.confidence == 0.0
    assert empty_pred.is_digit is False


# ---------------------------------------------------------------------------
# 2. PROB-18: CONSECUTIVE DIGIT & CURRENCY FORMATTING
# ---------------------------------------------------------------------------

def test_format_tokens_with_numbers_and_currency():
    """Contiguous single digits are grouped into multi-digit numbers and Indian Rupee prefixes."""
    # Multi-digit grouping
    tokens_pin = ["9", "8", "2", "0"]
    assert format_tokens_with_numbers_and_currency(tokens_pin) == ["9820"]

    # Indian Rupee prefix
    tokens_price = ["RUPEES", "5", "0", "0"]
    assert format_tokens_with_numbers_and_currency(tokens_price) == ["₹500"]

    # Mixed sentence with currency and words
    mixed_tokens = ["I", "WANT", "RUPEES", "2", "5", "0"]
    assert format_tokens_with_numbers_and_currency(mixed_tokens) == ["I", "WANT", "₹250"]

    # Verify integration with SentenceAccumulator
    acc = SentenceAccumulator()
    acc.add_token("RUPEES", token_type="phrase")
    acc.add_token("1", token_type="letter")
    acc.add_token("5", token_type="letter")
    acc.add_token("0", token_type="letter")

    # In sentence buffer, grouped as ₹150
    assert "₹150" in acc.sentence_buffer


# ---------------------------------------------------------------------------
# 3. PROB-24: HEALTHCARE & BANKING DOMAIN PACKS
# ---------------------------------------------------------------------------

def test_domain_packs_contents_and_translations():
    """Healthcare and banking packs provide translations across top Indic languages."""
    # Healthcare
    assert "CHEST PAIN" in HEALTHCARE_PACK
    assert HEALTHCARE_PACK["CHEST PAIN"].translations["hi"] == "सीने में दर्द"
    assert HEALTHCARE_PACK["DOCTOR"].translations["hi"] == "डॉक्टर"
    assert HEALTHCARE_PACK["WHERE HURTS"].translations["ta"] == "எங்கு வலிக்கிறது?"

    # Banking
    assert "AADHAAR" in BANKING_PACK
    assert BANKING_PACK["AADHAAR"].translations["hi"] == "आधार कार्ड"
    assert BANKING_PACK["ACCOUNT"].translations["te"] == "ఖాతా"
    assert BANKING_PACK["WITHDRAW"].translations["kn"] == "ಹಣ ಹಿಂಪಡೆಯುವುದು"


def test_domain_packs_manager_switching():
    """DomainPacksManager returns context-appropriate suggestions when switched."""
    # Switch to healthcare
    domain_packs_manager.set_domain("healthcare")
    hc_sugg = domain_packs_manager.get_domain_suggestions(limit=3)
    assert len(hc_sugg) == 3
    assert any("Pain" in s["display_name"] or "Fever" in s["display_name"] for s in hc_sugg)

    # Switch to banking
    domain_packs_manager.set_domain("banking")
    bk_sugg = domain_packs_manager.get_domain_suggestions(limit=3)
    assert len(bk_sugg) == 3
    assert any("Bank" in s["display_name"] or "Account" in s["display_name"] for s in bk_sugg)


def test_ws_set_domain_control_command(client):
    """Client switches domain via WebSocket control text."""
    with client.websocket_connect("/api/v1/ws/recognize") as ws:
        ws.send_text(json.dumps({"action": "set_domain", "domain": "healthcare"}))
        r_hc = ws.receive_json()
        assert r_hc["type"] == "info"
        assert r_hc["active_domain"] == "healthcare"
        assert len(r_hc["domain_suggestions"]) > 0

        ws.send_text(json.dumps({"action": "set_domain", "domain": "banking"}))
        r_bk = ws.receive_json()
        assert r_bk["type"] == "info"
        assert r_bk["active_domain"] == "banking"
