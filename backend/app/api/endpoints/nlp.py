"""
SETU — Natural Language Processing & Sign Language Translation (SLT) API.

Provides endpoints for:
1. Fingerspelling lexical & phonological auto-correction.
2. Multilingual ISL Gloss-to-Text translation (English, Hindi, Tamil, Telugu, Kannada).
3. Reverse English-to-ISL Gloss synthesis for bi-directional communication.
4. Linguistic syntax analysis and NMM (Non-Manual Marker) grammatical inflection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.nlp import (
    ISLGlossParser,
    indic_transformer,
    syntax_transformer,
    spell_corrector,
    neural_translator,
)

logger = logging.getLogger("setu.backend.api.nlp")

router = APIRouter(prefix="/v1/nlp", tags=["nlp"])

_parser = ISLGlossParser()

SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "native": "English", "alignment": "SVO"},
    {"code": "hi", "name": "Hindi", "native": "हिन्दी", "alignment": "SOV"},
    {"code": "ta", "name": "Tamil", "native": "தமிழ்", "alignment": "SOV"},
    {"code": "te", "name": "Telugu", "native": "తెలుగు", "alignment": "SOV"},
    {"code": "kn", "name": "Kannada", "native": "ಕನ್ನಡ", "alignment": "SOV"},
]


class SpellCorrectionRequest(BaseModel):
    text: str = Field(..., description="Fingerspelled string or word (e.g. 'THAMK')")
    max_cost: float = Field(1.4, description="Maximum phonological Levenshtein cost threshold")


class SpellCorrectionResponse(BaseModel):
    original: str
    corrected: str
    modified: bool


class TranslateRequest(BaseModel):
    tokens: Optional[List[str]] = Field(None, description="Sequence of ISL gloss tokens (e.g. ['ME', 'WATER', 'DRINK'])")
    text: Optional[str] = Field(None, description="Space-separated ISL gloss string (e.g. 'ME WATER DRINK')")
    target_languages: Optional[List[str]] = Field(
        default=["en", "hi", "ta", "te", "kn"],
        description="Target language codes to generate"
    )
    is_question: bool = Field(False, description="Whether non-manual facial marker indicates a question")
    is_negation: bool = Field(False, description="Whether non-manual head nod/shake indicates negation")


class TranslateResponse(BaseModel):
    input_tokens: List[str]
    translations: Dict[str, str]
    canonical_match: bool
    fluent_english: str
    rule_applied: Optional[str] = None


class ReverseGlossRequest(BaseModel):
    text: str = Field(..., description="English sentence to convert to ISL gloss tokens (e.g. 'What is your name?')")


class ReverseGlossResponse(BaseModel):
    input_text: str
    gloss_tokens: List[str]
    gloss_string: str


@router.get("/languages")
async def get_supported_languages() -> Dict[str, Any]:
    """Returns the list of supported spoken/written languages for ISL translation."""
    return {"languages": SUPPORTED_LANGUAGES}


@router.get("/status")
async def get_nlp_status() -> Dict[str, Any]:
    """Returns readiness and diagnostics for all NLP translation subsystems."""
    return {
        "status": "ready",
        "spell_corrector": {
            "dictionary_size": len(spell_corrector.dictionary),
            "phonological_confusion_pairs": len(spell_corrector.weighted_distance.__code__.co_consts),
        },
        "syntax_transformer": "ready",
        "indic_transformer": {
            "supported_languages": ["hi", "ta", "te", "kn"],
            "direct_sov_alignment": True,
        },
        "neural_translator": {
            "is_loaded": getattr(neural_translator, "is_loaded", False),
            "device": getattr(neural_translator, "device", "cpu"),
            "model_id": getattr(neural_translator, "model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
        },
    }


@router.post("/correct", response_model=SpellCorrectionResponse)
async def correct_spelling(request: SpellCorrectionRequest) -> SpellCorrectionResponse:
    """Auto-corrects fingerspelled words using phonological Levenshtein costs."""
    clean = request.text.strip().upper()
    if not clean:
        return SpellCorrectionResponse(original="", corrected="", modified=False)

    corrected = spell_corrector.correct(clean, max_cost_threshold=request.max_cost)
    return SpellCorrectionResponse(
        original=clean,
        corrected=corrected,
        modified=(clean != corrected),
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate_gloss(request: TranslateRequest) -> TranslateResponse:
    """Translates ISL gloss tokens into English (SVO) and Indic languages (SOV)."""
    raw_tokens: List[str] = []
    if request.tokens:
        raw_tokens = [t.strip().upper() for t in request.tokens if t.strip()]
    elif request.text:
        raw_tokens = [t.strip().upper() for t in request.text.strip().split() if t.strip()]

    if not raw_tokens:
        return TranslateResponse(
            input_tokens=[],
            translations={"en": ""},
            canonical_match=False,
            fluent_english="",
        )

    # 1. Lexical auto-correction pass on single-character runs (fingerspelling)
    clean_tokens: List[str] = []
    i = 0
    while i < len(raw_tokens):
        if len(raw_tokens[i]) == 1:
            run: List[str] = []
            while i < len(raw_tokens) and len(raw_tokens[i]) == 1:
                run.append(raw_tokens[i])
                i += 1
            word = "".join(run)
            clean_tokens.append(spell_corrector.correct(word))
        else:
            clean_tokens.append(raw_tokens[i])
            i += 1

    # 2. Exact Canonical Rule Match
    rule_match = _parser.match_rule(clean_tokens)
    canonical = rule_match is not None

    # 3. Syntax transformation (SOV -> SVO English)
    english_parsed = rule_match or syntax_transformer.transform(clean_tokens) or _parser.parse_gloss(clean_tokens)

    # 4. Indic Multilingual Translation
    indic_translations = indic_transformer.translate_all(clean_tokens)

    translations: Dict[str, str] = {"en": english_parsed}
    for lang, trans in indic_translations.items():
        if not request.target_languages or lang in request.target_languages:
            translations[lang] = trans

    # 5. Non-Manual Markers (NMMs) Grammatical Modifiers
    if request.is_negation:
        if "not" not in translations["en"].lower() and "no" not in translations["en"].lower():
            translations["en"] = f"Not {translations['en'].lstrip()}"
        for lang, neg_sfx in [("hi", " नहीं"), ("ta", " இல்லை"), ("te", " లేదు"), ("kn", " ಇಲ್ಲ")]:
            if lang in translations and neg_sfx.strip() not in translations[lang]:
                translations[lang] = translations[lang].rstrip("।.").strip() + neg_sfx

    if request.is_question:
        if not translations["en"].endswith("?"):
            translations["en"] = translations["en"].rstrip(".").strip() + "?"
        for lang in translations:
            if lang != "en" and not translations[lang].endswith("?"):
                translations[lang] = translations[lang].rstrip("।.").strip() + "?"

    return TranslateResponse(
        input_tokens=clean_tokens,
        translations=translations,
        canonical_match=canonical,
        fluent_english=translations["en"],
        rule_applied="canonical_idiom" if canonical else "syntax_transformer",
    )


@router.post("/to-gloss", response_model=ReverseGlossResponse)
async def english_to_isl_gloss(request: ReverseGlossRequest) -> ReverseGlossResponse:
    """
    Transforms English text into Indian Sign Language (ISL) gloss tokens.
    Converts English SVO structures into ISL SOV syntax and strips tense/articles.
    Prepares for bi-directional avatars or ISL generation.
    """
    import re
    text = request.text.strip()
    if not text:
        return ReverseGlossResponse(input_text="", gloss_tokens=[], gloss_string="")

    # Remove punctuation for tokenization
    clean = re.sub(r"[^\w\s]", "", text).upper()
    words = clean.split()

    # Common English stopwords not signed in ISL (articles, copulas)
    ISL_OMIT = {"A", "AN", "THE", "IS", "AM", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING", "TO", "OF"}
    
    # Simple SVO -> SOV rearrangement for common phrases
    filtered = [w for w in words if w not in ISL_OMIT]

    # Rule-based alignments for questions (Wh-word moves to end in ISL)
    wh_words = [w for w in filtered if w in ("WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH")]
    non_wh = [w for w in filtered if w not in ("WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH")]

    if wh_words:
        # Move Wh-word to final position
        gloss_tokens = non_wh + wh_words
    else:
        # Standard SOV heuristic: If Subject-Verb-Object pattern detected
        if len(filtered) >= 3 and filtered[0] in ("I", "YOU", "HE", "SHE", "WE", "THEY"):
            subj = filtered[0]
            verb = filtered[1]
            obj = filtered[2:]
            gloss_tokens = [subj] + obj + [verb]
        else:
            gloss_tokens = filtered

    return ReverseGlossResponse(
        input_text=text,
        gloss_tokens=gloss_tokens,
        gloss_string=" ".join(gloss_tokens),
    )
