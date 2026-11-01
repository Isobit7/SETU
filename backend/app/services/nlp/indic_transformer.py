"""
SETU — Multilingual Indic Language Translation Transformer (PROB-17).

Translates Indian Sign Language (ISL) gloss sequences directly into major
Indian languages:
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)

Linguistic Alignment:
Unlike English (SVO), Indian Sign Language and Indic languages naturally share
Subject-Object-Verb (SOV) structure with clause-final question particles and
postpositional markers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


# Canonical exact sentence mappings across Indic languages
CANONICAL_INDIC_IDIOMS: Dict[str, Dict[str, str]] = {
    # Greetings & Common Social Signs
    "NAMASTE": {
        "hi": "नमस्ते।",
        "ta": "வணக்கம்.",
        "te": "నమస్కారం.",
        "kn": "ನಮಸ್ಕಾರ.",
    },
    "HELLO": {
        "hi": "नमस्ते!",
        "ta": "வணக்கம்!",
        "te": "హలో / నమస్కారం!",
        "kn": "ಹಲೋ / ನಮಸ್ಕಾರ!",
    },
    "GOOD MORNING": {
        "hi": "शुभ प्रभात।",
        "ta": "காலை வணக்கம்.",
        "te": "శుభోదయం.",
        "kn": "ಶುಭೋದಯ.",
    },
    "THANK YOU": {
        "hi": "धन्यवाद।",
        "ta": "நன்றி.",
        "te": "ధన్యవాదాలు.",
        "kn": "ಧನ್ಯವಾದಗಳು.",
    },
    "THANKYOU": {
        "hi": "धन्यवाद।",
        "ta": "நன்றி.",
        "te": "ధన్యవాదాలు.",
        "kn": "ಧನ್ಯವಾದಗಳು.",
    },
    "PLEASE": {
        "hi": "कृपया।",
        "ta": "தயவுசெய்து.",
        "te": "దయచేసి.",
        "kn": "ದಯವಿಟ್ಟು.",
    },
    "SORRY": {
        "hi": "माफ़ कीजिए।",
        "ta": "மன்னிக்கவும்.",
        "te": "నన్ను క్షమించండి.",
        "kn": "ಕ್ಷಮಿಸಿ.",
    },
    "YES": {
        "hi": "हाँ।",
        "ta": "ஆம்.",
        "te": "అవును.",
        "kn": "ಹೌದು.",
    },
    "NO": {
        "hi": "नहीं।",
        "ta": "இல்லை.",
        "te": "కాదు.",
        "kn": "ಇಲ್ಲ.",
    },
    "WELCOME": {
        "hi": "स्वागत है।",
        "ta": "நல்வரவு.",
        "te": "స్వాగతం.",
        "kn": "ಸ್ವಾಗತ.",
    },

    # Common Conversational Sentences
    "YOU NAME WHAT": {
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?",
        "kn": "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
    },
    "YOUR NAME WHAT": {
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?",
        "kn": "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
    },
    "YOU LIVE WHERE": {
        "hi": "आप कहाँ रहते हैं?",
        "ta": "நீங்கள் எங்கே வசிக்கிறீர்கள்?",
        "te": "మీరు ఎక్కడ నివసిస్తున్నారు?",
        "kn": "ನೀವು ಎಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
    },
    "YOU COME WHEN": {
        "hi": "आप कब आएंगे?",
        "ta": "நீங்கள் எப்போது வருவீர்கள்?",
        "te": "మీరు ఎప్పుడు వస్తారు?",
        "kn": "ನೀವು ಯಾವಾಗ ಬರುತ್ತೀರಿ?",
    },
    "HOW YOU": {
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    },
    "HOW ARE YOU": {
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
    },
    "ME WATER WANT": {
        "hi": "मुझे पानी चाहिए।",
        "ta": "எனக்கு தண்ணீர் வேண்டும்.",
        "te": "నాకు నీళ్లు కావాలి.",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
    },
    "I WATER WANT": {
        "hi": "मुझे पानी चाहिए।",
        "ta": "எனக்கு தண்ணீர் வேண்டும்.",
        "te": "నాకు నీళ్లు కావాలి.",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
    },
    "I HELP NEED": {
        "hi": "मुझे मदद चाहिए।",
        "ta": "எனக்கு உதவி தேவை.",
        "te": "నాకు సహాయం కావాలి.",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು.",
    },
    "HELP NEED": {
        "hi": "कृपया मदद कीजिए।",
        "ta": "உதவி தேவை.",
        "te": "సహాయం కావాలి.",
        "kn": "ಸಹಾಯ ಬೇಕು.",
    },
    "DOCTOR WHERE": {
        "hi": "डॉक्टर कहाँ हैं?",
        "ta": "மருத்துவர் எங்கே?",
        "te": "డాక్టర్ ఎక్కడ ఉన్నారు?",
        "kn": "ವೈದ್ಯರು ಎಲ್ಲಿದ್ದಾರೆ?",
    },
    "HOSPITAL WHERE": {
        "hi": "अस्पताल कहाँ है?",
        "ta": "மருத்துவமனை எங்கே?",
        "te": "ఆసుపత్రి ఎక్కడ ఉంది?",
        "kn": "ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?",
    },
    "I TEA LIKE NOT": {
        "hi": "मुझे चाय पसंद नहीं है।",
        "ta": "எனக்கு தேநீர் பிடிக்காது.",
        "te": "నాకు టీ ఇష్టం లేదు.",
        "kn": "ನನಗೆ ಚಹಾ ಇಷ್ಟವಿಲ್ಲ.",
    },
    "I HAPPY": {
        "hi": "मैं खुश हूँ।",
        "ta": "நான் மகிழ்ச்சியாக இருக்கிறேன்.",
        "te": "నేను సంతోషంగా ఉన్నాను.",
        "kn": "ನಾನು ಸಂತೋಷವಾಗಿದ್ದೇನೆ.",
    },
}

# Individual Token Gloss Lexicon for SOV Morphological Composition
INDIC_LEXICON: Dict[str, Dict[str, str]] = {
    # Pronouns
    "ME": {"hi": "मुझे", "ta": "எனக்கு", "te": "నాకు", "kn": "ನನಗೆ"},
    "I": {"hi": "मैं", "ta": "நான்", "te": "నేను", "kn": "ನಾನು"},
    "YOU": {"hi": "आप", "ta": "நீங்கள்", "te": "మీరు", "kn": "ನೀವು"},
    "HE": {"hi": "वह", "ta": "அவர்", "te": "అతను", "kn": "ಅವನು"},
    "SHE": {"hi": "वह", "ta": "அவள்", "te": "ఆమె", "kn": "ಅವಳು"},
    "WE": {"hi": "हम", "ta": "நாங்கள்", "te": "మేము", "kn": "ನಾವು"},
    "THEY": {"hi": "वे", "ta": "அவர்கள்", "te": "వారు", "kn": "ಅವರು"},
    "MY": {"hi": "मेरा", "ta": "என்", "te": "నా", "kn": "ನನ್ನ"},
    "YOUR": {"hi": "आपका", "ta": "உங்கள்", "te": "మీ", "kn": "ನಿಮ್ಮ"},

    # Nouns
    "WATER": {"hi": "पानी", "ta": "தண்ணீர்", "te": "నీళ్లు", "kn": "ನೀರು"},
    "TEA": {"hi": "चाय", "ta": "தேநீர்", "te": "టీ", "kn": "ಚಹಾ"},
    "FOOD": {"hi": "खाना", "ta": "உணவு", "te": "ఆహారం", "kn": "ಊಟ"},
    "BOOK": {"hi": "किताब", "ta": "புத்தகம்", "te": "పుస్తకం", "kn": "ಪುಸ್ತಕ"},
    "DOCTOR": {"hi": "डॉक्टर", "ta": "மருத்துவர்", "te": "డాక్టర్", "kn": "ವೈದ್ಯರು"},
    "HOSPITAL": {"hi": "अस्पताल", "ta": "மருத்துவமனை", "te": "ఆసుపత్రి", "kn": "ಆಸ್ಪತ್ರೆ"},
    "HELP": {"hi": "मदद", "ta": "உதவி", "te": "సహాయం", "kn": "ಸಹಾಯ"},
    "NAME": {"hi": "नाम", "ta": "பெயர்", "te": "పేరు", "kn": "ಹೆಸರು"},
    "MONEY": {"hi": "पैसे", "ta": "பணம்", "te": "డబ్బులు", "kn": "ಹಣ"},
    "TIME": {"hi": "समय", "ta": "நேரம்", "te": "సమయం", "kn": "ಸಮಯ"},
    "FAMILY": {"hi": "परिवार", "ta": "குடும்பம்", "te": "కుటుంబం", "kn": "ಕುಟುಂಬ"},
    "HOUSE": {"hi": "घर", "ta": "வீடு", "te": "ఇల్లు", "kn": "ಮನೆ"},
    "FRIEND": {"hi": "दोस्त", "ta": "நண்பர்", "te": "స్నేహితుడు", "kn": "ಸ್ನೇಹಿತ"},

    # Verbs
    "WANT": {"hi": "चाहिए", "ta": "வேண்டும்", "te": "కావాలి", "kn": "ಬೇಕು"},
    "NEED": {"hi": "जरूरत है", "ta": "தேவை", "te": "అవసరం", "kn": "ಅಗತ್ಯವಿದೆ"},
    "LIKE": {"hi": "पसंद है", "ta": "பிடிக்கும்", "te": "ఇష్టం", "kn": "ಇಷ್ಟ"},
    "EAT": {"hi": "खाना", "ta": "சாப்பிடு", "te": "తిను", "kn": "ತಿನ್ನು"},
    "DRINK": {"hi": "पीना", "ta": "குடி", "te": "తాగు", "kn": "ಕುಡಿ"},
    "GO": {"hi": "जाना", "ta": "போ", "te": "వెళ్ళు", "kn": "ಹೋಗು"},
    "COME": {"hi": "आना", "ta": "வா", "te": "రా", "kn": "ಬಾ"},
    "SEE": {"hi": "देखना", "ta": "பார்", "te": "చూడు", "kn": "ನೋಡು"},
    "UNDERSTAND": {"hi": "समझ गया", "ta": "புரிந்தது", "te": "అర్థమైంది", "kn": "ಅರ್ಥವಾಯಿತು"},

    # Adjectives & Status
    "GOOD": {"hi": "अच्छा", "ta": "நல்லது", "te": "మంచిది", "kn": "ಒಳ್ಳೆಯದು"},
    "BAD": {"hi": "खराब", "ta": "மோசமானது", "te": "చెడ్డది", "kn": "ಕೆಟ್ಟದು"},
    "HAPPY": {"hi": "खुश", "ta": "மகிழ்ச்சி", "te": "సంతోషం", "kn": "ಸಂತೋಷ"},
    "SAD": {"hi": "दुखी", "ta": "வருத்தம்", "te": "బాధ", "kn": "ದುಃಖ"},

    # Wh-words
    "WHAT": {"hi": "क्या", "ta": "என்ன", "te": "ఏమిటి", "kn": "ಏನು"},
    "WHERE": {"hi": "कहाँ", "ta": "எங்கே", "te": "ఎక్కడ", "kn": "ಎಲ್ಲಿ"},
    "WHEN": {"hi": "कब", "ta": "எப்போது", "te": "ఎప్పుడు", "kn": "ಯಾವಾಗ"},
    "WHY": {"hi": "क्यों", "ta": "ஏன்", "te": "ఎందుకు", "kn": "ಏಕೆ"},
    "HOW": {"hi": "कैसे", "ta": "எப்படி", "te": "ఎలా", "kn": "ಹೇಗೆ"},

    # Negation
    "NOT": {"hi": "नहीं", "ta": "இல்லை", "te": "కాదు", "kn": "ಇಲ್ಲ"},
    "NO": {"hi": "नहीं", "ta": "இல்லை", "te": "కాదు", "kn": "ಇಲ್ಲ"},
}


class IndicTransformer:
    """Transforms ISL gloss sequences into fluent Hindi, Tamil, Telugu, and Kannada."""

    SUPPORTED_LANGUAGES = ("hi", "ta", "te", "kn")

    def translate(self, tokens: Sequence[str], lang: str = "hi") -> str:
        """Translates an ISL gloss token sequence into the specified Indic language."""
        if not tokens:
            return ""

        clean_tokens = [str(t).strip().upper() for t in tokens if str(t).strip()]
        if not clean_tokens:
            return ""

        # Normalize language code (e.g. 'hi-IN' -> 'hi')
        norm_lang = lang.split("-")[0].lower()
        if norm_lang not in self.SUPPORTED_LANGUAGES:
            norm_lang = "hi"

        sentence_key = " ".join(clean_tokens)

        # 1. Exact canonical idiom match
        if sentence_key in CANONICAL_INDIC_IDIOMS:
            idiom_dict = CANONICAL_INDIC_IDIOMS[sentence_key]
            if norm_lang in idiom_dict:
                return idiom_dict[norm_lang]

        # Check single-word exact match
        if len(clean_tokens) == 1 and clean_tokens[0] in CANONICAL_INDIC_IDIOMS:
            idiom_dict = CANONICAL_INDIC_IDIOMS[clean_tokens[0]]
            if norm_lang in idiom_dict:
                return idiom_dict[norm_lang]

        # 2. Compositional SOV Word Translation
        # In ISL and Indic languages, word order is naturally Subject -> Object -> Verb
        translated_words: List[str] = []
        is_question = any(t in ("WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH") for t in clean_tokens)
        has_negation = any(t in ("NOT", "NO", "NEVER") for t in clean_tokens)

        for token in clean_tokens:
            if token in ("NOT", "NO") and has_negation:
                continue  # Scoped at end for Indic verbs

            if token in INDIC_LEXICON and norm_lang in INDIC_LEXICON[token]:
                translated_words.append(INDIC_LEXICON[token][norm_lang])
            else:
                # Keep fingerspelled letters or unknown words as-is
                translated_words.append(token)

        # Append negation if present
        if has_negation and "NOT" in INDIC_LEXICON:
            translated_words.append(INDIC_LEXICON["NOT"][norm_lang])

        result = " ".join(translated_words)

        # Add appropriate punctuation
        if is_question:
            if not result.endswith("?"):
                result += "?"
        else:
            if norm_lang == "hi" and not result.endswith(("।", ".")):
                result += "।"
            elif not result.endswith((".", "।")):
                result += "."

        return result

    def translate_all(self, tokens: Sequence[str]) -> Dict[str, str]:
        """Returns translations for all supported Indic languages simultaneously."""
        return {lang: self.translate(tokens, lang=lang) for lang in self.SUPPORTED_LANGUAGES}


# Singleton instance
indic_transformer = IndicTransformer()

