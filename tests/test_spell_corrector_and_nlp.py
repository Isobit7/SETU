"""
SETU — Unit Tests for ISL Phonological Spell-Corrector, Accumulator Integration,
Geometric Tie-Breaker Heuristics, and NLP SLT Endpoints.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.nlp.spell_corrector import ISLSpellCorrector, spell_corrector
from backend.app.services.nlp.accumulator import SentenceAccumulator
from training.features.geometry import evaluate_alphabet_heuristics, compute_joint_angles_15


class MockParser:
    def match_rule(self, tokens):
        if tokens == ["ME", "WATER", "DRINK"]:
            return "I want water"
        return None

    def parse_gloss(self, tokens):
        return " ".join(tokens)


def test_spell_corrector_exact_match():
    assert spell_corrector.correct("HELLO") == "HELLO"
    assert spell_corrector.correct("THANK") == "THANK"
    assert spell_corrector.correct("WATER") == "WATER"
    assert spell_corrector.correct("DOCTOR") == "DOCTOR"


def test_spell_corrector_phonological_repair():
    # ISL MediaPipe confusion pairs: M<->N, single-letter typos, omissions
    assert spell_corrector.correct("THAMK") == "THANK"
    assert spell_corrector.correct("HELO") == "HELLO"
    assert spell_corrector.correct("WATR") == "WATER"


def test_spell_corrector_preserves_proper_names():
    # Should NOT warp personal names that exceed distance threshold
    assert spell_corrector.correct("SATHWIK") == "SATHWIK"
    assert spell_corrector.correct("AMRITA") == "AMRITA"


def test_accumulator_word_commit_spell_correction():
    acc = SentenceAccumulator(parser=MockParser(), word_timeout=0.05, sentence_timeout=0.20)
    
    # Simulate fingerspelling T - H - A - M - K
    for char in ["T", "H", "A", "M", "K"]:
        acc.add_token(char, token_type="letter", timestamp=100.0)
    
    assert acc.current_word == "THAMK"
    
    # Check timeout after pause
    result = acc.check_timeouts(current_time=100.10)
    assert result["word_committed"] is True
    # The committed token in sentence_tokens should be corrected to THANK
    assert "THANK" in acc.sentence_tokens
    assert acc.current_word == ""


def test_geometric_tie_breaker_m_vs_n():
    # Synthetic 21x3 hand landmarks
    pts = np.zeros((21, 3), dtype=np.float32)
    # Set wrist and MCPs to form palm
    pts[0] = [0.0, 0.0, 0.0]
    pts[9] = [0.0, 0.5, 0.0]  # Middle MCP
    
    # Fill remaining points to pass non-zero check
    for i in range(1, 21):
        pts[i] = [0.02 * i, 0.03 * i, 0.0]

    # Evaluate heuristic for candidate pair M vs N
    res = evaluate_alphabet_heuristics(pts, candidate_pair={"M", "N"})
    assert res in ("M", "N")


def test_geometric_tie_breaker_o_vs_c():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]
    pts[9] = [0.0, 0.5, 0.0]
    for i in range(1, 21):
        pts[i] = [0.02 * i, 0.03 * i, 0.0]

    # Thumb tip (4) and Index tip (8) close together -> O
    pts[4] = [0.1, 0.3, 0.0]
    pts[8] = [0.12, 0.3, 0.0]  # Very small aperture

    res = evaluate_alphabet_heuristics(pts, candidate_pair={"O", "C"})
    assert res == "O"


def test_nlp_api_endpoints():
    client = TestClient(app)

    # 1. Languages
    res_lang = client.get("/api/v1/nlp/languages")
    assert res_lang.status_code == 200
    langs = res_lang.json()["languages"]
    assert any(l["code"] == "hi" for l in langs)
    assert any(l["code"] == "en" for l in langs)

    # 2. Status
    res_status = client.get("/api/v1/nlp/status")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "ready"

    # 3. Spelling auto-correction
    res_corr = client.post("/api/v1/nlp/correct", json={"text": "THAMK"})
    assert res_corr.status_code == 200
    corr_data = res_corr.json()
    assert corr_data["original"] == "THAMK"
    assert corr_data["corrected"] == "THANK"
    assert corr_data["modified"] is True

    # 4. Translation (SOV to SVO + Indic)
    res_trans = client.post(
        "/api/v1/nlp/translate",
        json={"tokens": ["ME", "WATER", "DRINK"], "target_languages": ["en", "hi"]}
    )
    assert res_trans.status_code == 200
    trans_data = res_trans.json()
    assert trans_data["translations"]["en"] == "I want water"
    assert "hi" in trans_data["translations"]

    # 5. Reverse English-to-Gloss synthesis
    res_gloss = client.post("/api/v1/nlp/to-gloss", json={"text": "I want water"})
    assert res_gloss.status_code == 200
    gloss_data = res_gloss.json()
    assert "WATER" in gloss_data["gloss_tokens"]
    assert "WANT" in gloss_data["gloss_tokens"]
