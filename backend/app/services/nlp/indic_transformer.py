"""
SETU — Multilingual Indic Language Translation Transformer (PROB-17).

Translates Indian Sign Language (ISL) gloss sequences directly into major
Indian languages:
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)

Linguistic Alignment:
Unlike English (SVO), Indian Sign Language and Indic languages naturally share
Subject-Object-Verb (SOV) structure with clause-final question particles and
postpositional markers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


# Canonical exact sentence mappings across Indic languages
CANONICAL_INDIC_IDIOMS: Dict[str, Dict[str, str]] = {
    # Dynamic Master Animals & Nature
    "CAT": {"hi": "बिल्ली।", "ta": "பூனை.", "te": "పిల్లి.", "kn": "ಬೆಕ್ಕು.", "ml": "പൂച്ച."},
    "DOG": {"hi": "कुत्ता।", "ta": "நாய்.", "te": "కుక్క.", "kn": "ನಾಯಿ.", "ml": "നായ."},
    "BIRD": {"hi": "पक्षी।", "ta": "பறவை.", "te": "పక్షి.", "kn": "ಹಕ್ಕಿ / ಪಕ್ಷಿ.", "ml": "പക്ഷി."},
    "COW": {"hi": "गाय।", "ta": "பசு.", "te": "ఆవు.", "kn": "ಹಸು.", "ml": "പശു."},
    "FISH": {"hi": "मछली।", "ta": "மீன்.", "te": "చేప.", "kn": "ಮೀನು.", "ml": "മീൻ."},
    "MONSOON": {"hi": "मानसून।", "ta": "மழைக்காலம்.", "te": "వర్షాకాలం.", "kn": "ಮಳೆಗಾಲ.", "ml": "മഴക്കാലം."},

    # Dynamic Master Relations & People
    "FATHER": {"hi": "पिताजी।", "ta": "அப்பா.", "te": "నాన్న.", "kn": "ಅಪ್ಪ / ತಂದೆ.", "ml": "അച്ഛൻ."},
    "MOTHER": {"hi": "माँ।", "ta": "அம்மா.", "te": "அమ్మ.", "kn": "ಅಮ್ಮ / ತಾಯಿ.", "ml": "അമ്മ."},
    "BROTHER": {"hi": "भाई।", "ta": "சகோதரன்.", "te": "సోదరుడు.", "kn": "ಸಹೋದರ / ಅಣ್ಣ.", "ml": "സഹോദരൻ."},
    "SISTER": {"hi": "बहन।", "ta": "சகோதரி.", "te": "సోదరి.", "kn": "ಸಹೋದರಿ / ಅಕ್ಕ.", "ml": "സഹോദരി."},
    "TEACHER": {"hi": "अध्यापक।", "ta": "ஆசிரியர்.", "te": "ఉపాధ్యాయుడు.", "kn": "ಶಿಕ್ಷಕ.", "ml": "അധ്യാപകൻ."},
    "STUDENT": {"hi": "छात्र।", "ta": "மாணவர்.", "te": "విద్యార్థಿ.", "kn": "ವಿದ್ಯಾರ್ಥಿ.", "ml": "വിദ്യാർത്ഥി."},
    "BOY": {"hi": "लड़का।", "ta": "சிறுவன்.", "te": "அబ్బాయి.", "kn": "ಹುಡುಗ.", "ml": "ആൺകുട്ടി."},
    "GIRL": {"hi": "लड़की।", "ta": "சிறுமி.", "te": "అమ్మాయి.", "kn": "ಹುಡುಗಿ.", "ml": "പെൺകുട്ടി."},

    # Dynamic Master Actions & Objects
    "SLEEP": {"hi": "सोना।", "ta": "தூங்கு.", "te": "పడుకో.", "kn": "ಮಲಗು.", "ml": "ഉറങ്ങുക."},
    "SIT": {"hi": "बैठना।", "ta": "உட்காரு.", "te": "కూర్చో.", "kn": "ಕುಳಿತುಕೋ.", "ml": "ಇരിക്കുക."},
    "STAND": {"hi": "खड़े होना।", "ta": "நில்லு.", "te": "నిలబడు.", "kn": "ನಿಂತುಕೋ.", "ml": "നിൽക്കുക."},
    "STOP": {"hi": "रुको।", "ta": "நில்.", "te": "ఆగు.", "kn": "ನಿಲ್ಲು.", "ml": "നിർത്തുക."},
    "WAIT": {"hi": "इंतज़ार करो।", "ta": "காத்திரு.", "te": "వేచి ఉండు.", "kn": "ಕಾಯಿ.", "ml": "ಕಾത്തിരിക്കുക."},
    "STUDY": {"hi": "पढ़ाई।", "ta": "படிப்பு.", "te": "చదువు.", "kn": "ಓದು / ಅಧ್ಯಯನ.", "ml": "പഠനം."},
    "WORK": {"hi": "काम।", "ta": "வேலை.", "te": "పని.", "kn": "ಕೆಲಸ.", "ml": "ജോലി."},
    "WRITE": {"hi": "लिखना।", "ta": "எழுது.", "te": "రాయి.", "kn": "ಬರೆ.", "ml": "ಎഴുതുക."},
    "DRINKING": {"hi": "पीना।", "ta": "குடிக்க.", "te": "తాగు.", "kn": "ಕುಡಿ.", "ml": "കുടിക്കുക."},
    "UMBRELLA": {"hi": "छाता।", "ta": "குடை.", "te": "గొడుగు.", "kn": "ಛತ್ರಿ.", "ml": "കുട."},
    "RING": {"hi": "अंगूठी।", "ta": "மோதிரம்.", "te": "உங்கరం.", "kn": "ಉಂಗುರ.", "ml": "മോതിരം."},
    "POWER": {"hi": "शक्ति।", "ta": "சக்தி.", "te": "శక్తి.", "kn": "ಶಕ್ತಿ.", "ml": "ശಕ್ತಿ."},
    "MUSIC": {"hi": "संगीत।", "ta": "இசை.", "te": "సంగీతం.", "kn": "ಸಂಗೀತ.", "ml": "സംഗീതം."},
    "NOSE": {"hi": "नाक।", "ta": "மூக்கு.", "te": "ముక్కు.", "kn": "ಮೂಗು.", "ml": "മൂക്ക്."},
    "TEETH": {"hi": "दाँत।", "ta": "பற்கள்.", "te": "పళ్ళు.", "kn": "ಹಲ್ಲುಗಳು.", "ml": "പല്ലുകൾ."},
    "HAND": {"hi": "हाथ।", "ta": "கை.", "te": "చేయి.", "kn": "ಕೈ.", "ml": "കൈ."},
    "BLACK": {"hi": "काला।", "ta": "கருப்பு.", "te": "నలుపు.", "kn": "ಕಪ್ಪು.", "ml": "കറുപ്പ്."},
    "GREY": {"hi": "धूसर।", "ta": "சாம்பல் நிறம்.", "te": "బూడిద రంగు.", "kn": "ಬೂದು ಬಣ್ಣ.", "ml": "ചാരനിറം."},
    "WHITE": {"hi": "सफ़ेद।", "ta": "வெள்ளை.", "te": "తెలుపు.", "kn": "ಬಿಳಿ.", "ml": "വെള്ള."},
    "ANGRY": {"hi": "गुस्सा।", "ta": "கோபம்.", "te": "కోపం.", "kn": "ಕೋಪ.", "ml": "ദേഷ്യം."},

    # Dynamic Master Time
    "TODAY": {"hi": "आज।", "ta": "இன்று.", "te": "ఈ రోజు.", "kn": "ಇವತ್ತು / ಇಂದು.", "ml": "ഇന്ന്."},
    "TOMORROW": {"hi": "कल।", "ta": "நாளை.", "te": "రేపు.", "kn": "ನಾಳೆ.", "ml": "நாളെ."},
    "YESTERDAY": {"hi": "कल।", "ta": "நேற்று.", "te": "నిన్న.", "kn": "ನಿನ್ನೆ.", "ml": "ഇന്നലെ."},
    "EVENING": {"hi": "शाम।", "ta": "மாலை.", "te": "సాయంత్రం.", "kn": "ಸಂಜೆ.", "ml": "വൈകുന്നೇരം."},
    "YOU ARE PERFECT": {"hi": "आप बहुत अच्छे हैं!", "ta": "நீங்கள் அருமையாக இருக்கிறீர்கள்!", "te": "మీరు చాలా బాగున్నారు!", "kn": "ನೀವು ಪರಿಪೂರ್ಣರಾಗಿದ್ದೀರಿ!", "ml": "നിങ്ങൾ മികച്ചയാളാണ്!"},

    # Greetings & Common Social Signs
    "NAMASTE": {
        "hi": "नमस्ते।",
        "ta": "வணக்கம்.",
        "te": "నమస్కారం.",
        "kn": "ನಮಸ್ಕಾರ.",
        "ml": "നമസ്കാരം.",
    },
    "HELLO": {
        "hi": "नमस्ते!",
        "ta": "வணக்கம்!",
        "te": "హలో / నమస్కారం!",
        "kn": "ಹಲೋ / ನಮಸ್ಕಾರ!",
        "ml": "ഹലോ / നമസ്കാരം!",
    },
    "GOOD MORNING": {
        "hi": "शुभ प्रभात।",
        "ta": "காலை வணக்கம்.",
        "te": "శుభోదయం.",
        "kn": "ಶುಭೋದಯ.",
        "ml": "സുപ്രഭാതം.",
    },
    "GOOD AFTERNOON": {
        "hi": "शुभ दोपहर।",
        "ta": "மதிய வணக்கம்.",
        "te": "శుభ మధ్యాహ్నం.",
        "kn": "ಶುಭ ಮಧ್ಯಾಹ್ನ.",
        "ml": "ശുഭ ഉച്ചതിരിഞ്ഞ്.",
    },
    "GOOD NIGHT": {
        "hi": "शुभ रात्रि।",
        "ta": "இனிய இரவு.",
        "te": "శుభరాత్రి.",
        "kn": "ಶುಭರಾತ್ರಿ.",
        "ml": "ശുഭരാത്രി.",
    },
    "THANK YOU": {
        "hi": "धन्यवाद।",
        "ta": "நன்றி.",
        "te": "ధన్యవాదాలు.",
        "kn": "ಧನ್ಯವಾದಗಳು.",
        "ml": "നന്ദി.",
    },
    "THANKYOU": {
        "hi": "धन्यवाद।",
        "ta": "நன்றி.",
        "te": "ధన్యవాదాలు.",
        "kn": "ಧನ್ಯವಾದಗಳು.",
        "ml": "നന്ദി.",
    },
    "PLEASE": {
        "hi": "कृपया।",
        "ta": "தயவுசெய்து.",
        "te": "దయచేసి.",
        "kn": "ದಯವಿಟ್ಟು.",
        "ml": "ദയവായി.",
    },
    "SORRY": {
        "hi": "माफ़ कीजिए।",
        "ta": "மன்னிக்கவும்.",
        "te": "నన్ను క్షమించండి.",
        "kn": "ಕ್ಷಮಿಸಿ.",
        "ml": "ക്ഷമിക്കണം.",
    },
    "YES": {
        "hi": "हाँ।",
        "ta": "ஆம்.",
        "te": "అవును.",
        "kn": "ಹೌದು.",
        "ml": "അതെ.",
    },
    "NO": {
        "hi": "नहीं।",
        "ta": "இல்லை.",
        "te": "కాదు.",
        "kn": "ಇಲ್ಲ.",
        "ml": "അല്ല.",
    },
    "WELCOME": {
        "hi": "स्वागत है।",
        "ta": "நல்வரவு.",
        "te": "స్వాగతం.",
        "kn": "ಸ್ವಾಗತ.",
        "ml": "സ്വാഗതം.",
    },

    # Common Conversational Sentences
    "YOU NAME WHAT": {
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?",
        "kn": "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
        "ml": "നിങ്ങളുടെ പേര് എന്താണ്?",
    },
    "YOUR NAME WHAT": {
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?",
        "kn": "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
        "ml": "നിങ്ങളുടെ പേര് എന്താണ്?",
    },
    "NAME WHAT": {
        "hi": "आपका नाम क्या है?",
        "ta": "உங்கள் பெயர் என்ன?",
        "te": "మీ పేరు ఏమిటి?",
        "kn": "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
        "ml": "നിങ്ങളുടെ പേര് എന്താണ്?",
    },
    "YOU LIVE WHERE": {
        "hi": "आप कहाँ रहते हैं?",
        "ta": "நீங்கள் எங்கே வசிக்கிறீர்கள்?",
        "te": "మీరు ఎక్కడ నివసిస్తున్నారు?",
        "kn": "ನೀವು ಎಲ್ಲಿ ವಾಸಿಸುತ್ತೀರಿ?",
        "ml": "നിങ്ങൾ എവിടെയാണ് താമസിക്കുന്നത്?",
    },
    "YOU COME WHEN": {
        "hi": "आप कब आएंगे?",
        "ta": "நீங்கள் எப்போது வருவீர்கள்?",
        "te": "మీరు ఎప్పుడు వస్తారు?",
        "kn": "ನೀವು ಯಾವಾಗ ಬರುತ್ತೀರಿ?",
        "ml": "നിങ്ങൾ എപ്പോഴാണ് വരുന്നത്?",
    },
    "HOW YOU": {
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "നിങ്ങൾക്ക് സുഖമാണോ?",
    },
    "HOW ARE YOU": {
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "നിങ്ങൾക്ക് സുഖമാണോ?",
    },
    "YOU HOW": {
        "hi": "आप कैसे हैं?",
        "ta": "நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "te": "మీరు ఎలా ఉన్నారు?",
        "kn": "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ml": "നിങ്ങൾക്ക് സുഖമാണോ?",
    },
    "ME WATER WANT": {
        "hi": "मुझे पानी चाहिए।",
        "ta": "எனக்கு தண்ணீர் வேண்டும்.",
        "te": "నాకు నీళ్లు కావాలి.",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
        "ml": "എനിക്ക് വെള്ളം വേണം.",
    },
    "I WATER WANT": {
        "hi": "मुझे पानी चाहिए।",
        "ta": "எனக்கு தண்ணீர் வேண்டும்.",
        "te": "నాకు నీళ్లు కావాలి.",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
        "ml": "എനിക്ക് വെള്ളം വേണം.",
    },
    "WATER WANT": {
        "hi": "मुझे पानी चाहिए।",
        "ta": "எனக்கு தண்ணீர் வேண்டும்.",
        "te": "నాకు నీళ్లు కావాలి.",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು.",
        "ml": "എനിക്ക് വെള്ളം വേണം.",
    },
    "WATER": {
        "hi": "पानी।",
        "ta": "தண்ணீர்.",
        "te": "నీళ్లు.",
        "kn": "ನೀರು.",
        "ml": "വെള്ളം.",
    },
    "FOOD": {
        "hi": "खाना।",
        "ta": "உணவு.",
        "te": "ఆహారం.",
        "kn": "ಊಟ.",
        "ml": "ഭക്ഷണം.",
    },
    "EAT": {
        "hi": "खाना खाना।",
        "ta": "சாப்பிட வேண்டும்.",
        "te": "తినాలి.",
        "kn": "ತಿನ್ನಬೇಕು.",
        "ml": "കഴിക്കണം.",
    },
    "DRINK": {
        "hi": "पीना।",
        "ta": "குடிக்க வேண்டும்.",
        "te": "తాగాలి.",
        "kn": "ಕುಡಿಯಬೇಕು.",
        "ml": "കുടിക്കണം.",
    },
    "HELP": {
        "hi": "कृपया मदद कीजिए।",
        "ta": "உதவி தேவை.",
        "te": "సహాయం కావాలి.",
        "kn": "ಸಹಾಯ ಬೇಕು.",
        "ml": "എനിക്ക് സഹായം വേണം.",
    },
    "I HELP NEED": {
        "hi": "मुझे मदद चाहिए।",
        "ta": "எனக்கு உதவி தேவை.",
        "te": "నాకు సహాయం కావాలి.",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು.",
        "ml": "എനിക്ക് സഹായം വേണം.",
    },
    "ME HELP NEED": {
        "hi": "मुझे मदद चाहिए।",
        "ta": "எனக்கு உதவி தேவை.",
        "te": "నాకు సహాయం కావాలి.",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು.",
        "ml": "എനിക്ക് സഹായം വേണം.",
    },
    "HELP NEED": {
        "hi": "कृपया मदद कीजिए।",
        "ta": "உதவி தேவை.",
        "te": "సహాయం కావాలి.",
        "kn": "ಸಹಾಯ ಬೇಕು.",
        "ml": "എനിക്ക് സഹായം വേണം.",
    },
    "DOCTOR WHERE": {
        "hi": "डॉक्टर कहाँ हैं?",
        "ta": "மருத்துவர் எங்கே?",
        "te": "డాక్టర్ ఎక్కడ ఉన్నారు?",
        "kn": "ವೈದ್ಯರು ಎಲ್ಲಿದ್ದಾರೆ?",
        "ml": "ഡോക്ടർ എവിടെയാണ്?",
    },
    "HOSPITAL WHERE": {
        "hi": "अस्पताल कहाँ है?",
        "ta": "மருத்துவமனை எங்கே?",
        "te": "ఆసుపత్రి ఎక్కడ ఉంది?",
        "kn": "ಆಸ್ಪತ್ರೆ ಎಲ್ಲಿದೆ?",
        "ml": "ആശുപത്രി എവിടെയാണ്?",
    },
    "BATHROOM WHERE": {
        "hi": "शौचालय कहाँ है?",
        "ta": "கழிப்பறை எங்கே?",
        "te": "వాష్‌రూమ్ ఎక్కడ ఉంది?",
        "kn": "ಶೌಚಾಲಯ ಎಲ್ಲಿದೆ?",
        "ml": "ബാത്ത്റൂം എവിടെയാണ്?",
    },
    "TOILET WHERE": {
        "hi": "शौचालय कहाँ है?",
        "ta": "கழிப்பறை எங்கே?",
        "te": "వాష్‌రూమ్ ఎక్కడ ఉంది?",
        "kn": "ಶೌಚಾಲಯ ಎಲ್ಲಿದೆ?",
        "ml": "ബാത്ത്റൂം എവിടെയാണ്?",
    },
    "DOCTOR": {
        "hi": "डॉक्टर।",
        "ta": "மருத்துவர்.",
        "te": "డాక్టర్.",
        "kn": "ವೈದ್ಯರು.",
        "ml": "ഡോക്ടർ.",
    },
    "MEDICINE": {
        "hi": "दवा।",
        "ta": "மருந்து.",
        "te": "మందు.",
        "kn": "ಔಷಧಿ.",
        "ml": "മരുന്ന്.",
    },
    "I TEA LIKE NOT": {
        "hi": "मुझे चाय पसंद नहीं है।",
        "ta": "எனக்கு தேநீர் பிடிக்காது.",
        "te": "నాకు టీ ఇష్టం లేదు.",
        "kn": "ನನಗೆ ಚಹಾ ಇಷ್ಟವಿಲ್ಲ.",
        "ml": "എനിക്ക് ചായ ഇഷ്ടമല്ല.",
    },
    "I HAPPY": {
        "hi": "मैं खुश हूँ।",
        "ta": "நான் மகிழ்ச்சியாக இருக்கிறேன்.",
        "te": "నేను సంతోషంగా ఉన్నాను.",
        "kn": "ನಾನು ಸಂತೋಷವಾಗಿದ್ದೇನೆ.",
        "ml": "ഞാൻ സന്തോഷവാനാണ്.",
    },
    "GOOD": {
        "hi": "अच्छा।",
        "ta": "நல்லது.",
        "te": "మంచిది.",
        "kn": "ಒಳ್ಳೆಯದು.",
        "ml": "നല്ലത്.",
    },
    "BAD": {
        "hi": "खराब।",
        "ta": "மோசமானது.",
        "te": "చెడ్డది.",
        "kn": "ಕೆಟ್ಟದು.",
        "ml": "മോശം.",
    },
    "TIME WHAT": {
        "hi": "समय क्या हुआ है?",
        "ta": "நேரம் என்ன?",
        "te": "సమయం ఎంతైంది?",
        "kn": "ಸಮಯ ಎಷ್ಟು?",
        "ml": "സമയം എത്രയായി?",
    },
}

# Individual Token Gloss Lexicon for SOV Morphological Composition
INDIC_LEXICON: Dict[str, Dict[str, str]] = {
    # Pronouns
    "ME": {"hi": "मुझे", "ta": "எனக்கு", "te": "నాకు", "kn": "ನನಗೆ", "ml": "എനിക്ക്"},
    "I": {"hi": "मैं", "ta": "நான்", "te": "నేను", "kn": "ನಾನು", "ml": "ഞാൻ"},
    "YOU": {"hi": "आप", "ta": "நீங்கள்", "te": "మీరు", "kn": "ನೀವು", "ml": "നിങ്ങൾ"},
    "HE": {"hi": "वह", "ta": "அவர்", "te": "అతను", "kn": "ಅವನು", "ml": "അവൻ"},
    "SHE": {"hi": "वह", "ta": "அவள்", "te": "ఆమె", "kn": "ಅವಳು", "ml": "അവൾ"},
    "WE": {"hi": "हम", "ta": "நாங்கள்", "te": "మేము", "kn": "ನಾವು", "ml": "ഞങ്ങൾ"},
    "THEY": {"hi": "वे", "ta": "அவர்கள்", "te": "వారు", "kn": "ಅವರು", "ml": "അവർ"},
    "MY": {"hi": "मेरा", "ta": "என்", "te": "నా", "kn": "ನನ್ನ", "ml": "എന്റെ"},
    "YOUR": {"hi": "आपका", "ta": "உங்கள்", "te": "మీ", "kn": "ನಿಮ್ಮ", "ml": "നിങ്ങളുടെ"},

    # Nouns
    "WATER": {"hi": "पानी", "ta": "தண்ணீர்", "te": "నీళ్లు", "kn": "ನೀರು", "ml": "വെള്ളം"},
    "TEA": {"hi": "चाय", "ta": "தேநீர்", "te": "టీ", "kn": "ಚಹಾ", "ml": "ചായ"},
    "FOOD": {"hi": "खाना", "ta": "உணவு", "te": "ఆహారం", "kn": "ಊಟ", "ml": "ഭക്ഷണം"},
    "BOOK": {"hi": "किताब", "ta": "புத்தகம்", "te": "పుస్తకం", "kn": "ಪುಸ್ತಕ", "ml": "പുസ്തകം"},
    "DOCTOR": {"hi": "डॉक्टर", "ta": "மருத்துவர்", "te": "డాక్టర్", "kn": "ವೈದ್ಯರು", "ml": "ഡോക്ടർ"},
    "HOSPITAL": {"hi": "अस्पताल", "ta": "மருத்துவமனை", "te": "ఆసుపత్రి", "kn": "ಆಸ್ಪತ್ರೆ", "ml": "ആശുപത്രി"},
    "HELP": {"hi": "मदद", "ta": "உதவி", "te": "సహాయం", "kn": "ಸಹಾಯ", "ml": "സഹായം"},
    "NAME": {"hi": "नाम", "ta": "பெயர்", "te": "పేరు", "kn": "ಹೆಸರು", "ml": "പേര്"},
    "MONEY": {"hi": "पैसे", "ta": "பணம்", "te": "డబ్బులు", "kn": "ಹಣ", "ml": "പണം"},
    "TIME": {"hi": "समय", "ta": "நேரம்", "te": "సమయం", "kn": "ಸಮಯ", "ml": "സമയം"},
    "FAMILY": {"hi": "परिवार", "ta": "குடும்பம்", "te": "కుటుంబం", "kn": "ಕುಟುಂಬ", "ml": "കുടുംബം"},
    "HOUSE": {"hi": "घर", "ta": "வீடு", "te": "ఇల్లు", "kn": "ಮನೆ", "ml": "വീട്"},
    "FRIEND": {"hi": "दोस्त", "ta": "நண்பர்", "te": "స్నేహితుడు", "kn": "ಸ್ನೇಹಿತ", "ml": "സുഹൃത്ത്"},
    "MEDICINE": {"hi": "दवा", "ta": "மருந்து", "te": "మందు", "kn": "ಔಷಧಿ", "ml": "മരുന്ന്"},

    # Verbs
    "WANT": {"hi": "चाहिए", "ta": "வேண்டும்", "te": "కావాలి", "kn": "ಬೇಕು", "ml": "വേണം"},
    "NEED": {"hi": "जरूरत है", "ta": "தேவை", "te": "ಅವసరం", "kn": "ಅಗತ್ಯವಿದೆ", "ml": "ആവശ്യമാണ്"},
    "LIKE": {"hi": "पसंद है", "ta": "பிடிக்கும்", "te": "ఇష్టం", "kn": "ಇಷ್ಟ", "ml": "ഇഷ്ടമാണ്"},
    "EAT": {"hi": "खाना", "ta": "சாப்பிடு", "te": "తిను", "kn": "ತಿನ್ನು", "ml": "കഴിക്കുക"},
    "DRINK": {"hi": "पीना", "ta": "குடி", "te": "తాగు", "kn": "ಕುಡಿ", "ml": "കുടിക്കുക"},
    "GO": {"hi": "जाना", "ta": "போ", "te": "వెళ్ళు", "kn": "ಹೋಗು", "ml": "പോകുക"},
    "COME": {"hi": "आना", "ta": "வா", "te": "రా", "kn": "ಬಾ", "ml": "വരുക"},
    "SEE": {"hi": "देखना", "ta": "பார்", "te": "చూడు", "kn": "ನೋಡು", "ml": "കാണുക"},
    "UNDERSTAND": {"hi": "समझ गया", "ta": "புரிந்தது", "te": "అర్థమైంది", "kn": "ಅರ್ಥವಾಯಿತು", "ml": "മനസ്സിലായി"},

    # Adjectives & Status
    "GOOD": {"hi": "अच्छा", "ta": "நல்லது", "te": "మంచిది", "kn": "ಒಳ್ಳೆಯದು", "ml": "നല്ലത്"},
    "BAD": {"hi": "खराब", "ta": "மோசமானது", "te": "చెడ్డది", "kn": "ಕೆಟ್ಟದು", "ml": "മോശം"},
    "HAPPY": {"hi": "खुश", "ta": "மகிழ்ச்சி", "te": "సంతోషం", "kn": "ಸಂತೋಷ", "ml": "സന്തോഷം"},
    "SAD": {"hi": "दुखी", "ta": "வருத்தம்", "te": "బాధ", "kn": "ದುಃಖ", "ml": "വിഷമം"},

    # Dynamic Master Animals & Stems
    "CAT": {"hi": "बिल्ली", "ta": "பூனை", "te": "పిల్లి", "kn": "ಬೆಕ್ಕು", "ml": "പൂച്ച"},
    "DOG": {"hi": "कुत्ता", "ta": "நாய்", "te": "కుక్క", "kn": "ನಾಯಿ", "ml": "നായ"},
    "BIRD": {"hi": "पक्षी", "ta": "பறவை", "te": "పక్షి", "kn": "ಹಕ್ಕಿ", "ml": "പക്ഷി"},
    "COW": {"hi": "गाय", "ta": "பசு", "te": "ఆవు", "kn": "ಹಸು", "ml": "പശു"},
    "FISH": {"hi": "मछली", "ta": "மீன்", "te": "చేప", "kn": "ಮೀನು", "ml": "മീൻ"},
    "MONSOON": {"hi": "मानसून", "ta": "மழைக்காலம்", "te": "వర్షాకాలం", "kn": "ಮಳೆಗಾಲ", "ml": "മഴക്കാലം"},
    "FATHER": {"hi": "पिताजी", "ta": "அப்பா", "te": "నాన్న", "kn": "ಅಪ್ಪ", "ml": "അച്ഛൻ"},
    "MOTHER": {"hi": "माँ", "ta": "அம்மா", "te": "అమ్మ", "kn": "ಅಮ್ಮ", "ml": "അമ്മ"},
    "BROTHER": {"hi": "भाई", "ta": "சகோதரன்", "te": "సోదరుడు", "kn": "ಅಣ್ಣ / ತಮ್ಮ", "ml": "സഹോദരൻ"},
    "SISTER": {"hi": "बहन", "ta": "சகோதரி", "te": "సోదరి", "kn": "ಅಕ್ಕ / ತಂಗಿ", "ml": "சகோதரி"},
    "TEACHER": {"hi": "अध्यापक", "ta": "ஆசிரியர்", "te": "ఉపాధ్యాయుడు", "kn": "ಶಿಕ್ಷಕ", "ml": "അധ്യാപകൻ"},
    "STUDENT": {"hi": "छात्र", "ta": "மாணவர்", "te": "విద్యార్థಿ", "kn": "ವಿದ್ಯಾರ್ಥಿ", "ml": "വിദ്യാർത്ഥി"},
    "BOY": {"hi": "लड़का", "ta": "சிறுவன்", "te": "அబ్బాయి", "kn": "ಹುಡುಗ", "ml": "ആൺകുട്ടി"},
    "GIRL": {"hi": "लड़की", "ta": "சிறுமி", "te": "அమ్మాయి", "kn": "ಹುಡುಗಿ", "ml": "പെൺകുട്ടി"},
    "SLEEP": {"hi": "सोना", "ta": "தூங்கு", "te": "పడుకో", "kn": "ಮಲಗು", "ml": "ഉറങ്ങുക"},
    "SIT": {"hi": "बैठना", "ta": "உட்காரு", "te": "కూర్చో", "kn": "ಕುಳಿತುಕೋ", "ml": "ഇരിക്കുക"},
    "STAND": {"hi": "खड़े होना", "ta": "நில்லு", "te": "నిలబడు", "kn": "ನಿಂತುಕೋ", "ml": "നിൽക്കുക"},
    "STOP": {"hi": "रुको", "ta": "நில்", "te": "ఆగు", "kn": "ನಿಲ್ಲು", "ml": "നിർത്തുക"},
    "WAIT": {"hi": "इंतज़ार", "ta": "காத்திரு", "te": "వేచి ఉండు", "kn": "ಕಾಯಿ", "ml": "കാത്തിരിക്കുക"},
    "STUDY": {"hi": "पढ़ाई", "ta": "படிப்பு", "te": "చదువు", "kn": "ಓದು", "ml": "പഠനം"},
    "WORK": {"hi": "काम", "ta": "வேலை", "te": "పని", "kn": "ಕೆಲಸ", "ml": "ജോലി"},
    "WRITE": {"hi": "लिखना", "ta": "எழுது", "te": "రాయి", "kn": "ಬರೆ", "ml": "ಎഴുതുക"},
    "DRINKING": {"hi": "पीना", "ta": "குடிக்க", "te": "తాగు", "kn": "ಕುಡಿ", "ml": "കുടിക്കുക"},
    "UMBRELLA": {"hi": "छाता", "ta": "குடை", "te": "గొడుగు", "kn": "ಛತ್ರಿ", "ml": "കുട"},
    "RING": {"hi": "अंगूठी", "ta": "மோதிரம்", "te": "ఉంగరం", "kn": "ಉಂಗುರ", "ml": "మోതിരം"},
    "POWER": {"hi": "शक्ति", "ta": "சக்தி", "te": "శక్తి", "kn": "ಶಕ್ತಿ", "ml": "ശಕ್ತಿ"},
    "MUSIC": {"hi": "संगीत", "ta": "இசை", "te": "సంగీతం", "kn": "ಸಂಗೀತ", "ml": "ಸಂಗೀതം"},
    "NOSE": {"hi": "नाक", "ta": "மூக்கு", "te": "ముక్కు", "kn": "ಮೂಗು", "ml": "മൂക്ക്"},
    "TEETH": {"hi": "दाँत", "ta": "பற்கள்", "te": "పళ్ళు", "kn": "ಹಲ್ಲುಗಳು", "ml": "പല്ലുകൾ"},
    "HAND": {"hi": "हाथ", "ta": "கை", "te": "చేయి", "kn": "ಕೈ", "ml": "കೈ"},
    "BLACK": {"hi": "काला", "ta": "கருப்பு", "te": "నలుపు", "kn": "ಕಪ್ಪು", "ml": "കറുപ്പ്"},
    "GREY": {"hi": "धूसर", "ta": "சாம்பல் நிறம்", "te": "బూడిద రంగు", "kn": "ಬೂದು", "ml": "ചാരനിറം"},
    "WHITE": {"hi": "सफ़ेद", "ta": "வெள்ளை", "te": "తెలుపు", "kn": "ಬಿಳಿ", "ml": "വെള്ള"},
    "ANGRY": {"hi": "गुस्सा", "ta": "கோபம்", "te": "కోపం", "kn": "ಕೋಪ", "ml": "ദേഷ്യം"},
    "TODAY": {"hi": "आज", "ta": "இன்று", "te": "ఈ రోజు", "kn": "ಇವತ್ತು", "ml": "ಇന്ന്"},
    "TOMORROW": {"hi": "कल", "ta": "நாளை", "te": "రేపు", "kn": "ನಾಳೆ", "ml": "நாളെ"},
    "YESTERDAY": {"hi": "कल", "ta": "நேற்று", "te": "ನಿన్న", "kn": "ನಿನ್ನೆ", "ml": "ഇന്നലെ"},
    "EVENING": {"hi": "शाम", "ta": "மாலை", "te": "సాయంత్రం", "kn": "ಸಂಜೆ", "ml": "വൈകുന്നೇരം"},

    # Wh-words
    "WHAT": {"hi": "क्या", "ta": "என்ன", "te": "ఏమిటి", "kn": "ಏನು", "ml": "എന്താണ്"},
    "WHERE": {"hi": "कहाँ", "ta": "எங்கே", "te": "ఎక్కడ", "kn": "ಎಲ್ಲಿ", "ml": "എവിടെ"},
    "WHEN": {"hi": "कब", "ta": "எப்போது", "te": "ఎప్పుడు", "kn": "ಯಾವಾಗ", "ml": "എപ്പോൾ"},
    "WHY": {"hi": "क्यों", "ta": "ஏன்", "te": "ఎందుకు", "kn": "ಏಕೆ", "ml": "എന്തുകൊണ്ട്"},
    "HOW": {"hi": "कैसे", "ta": "எப்படி", "te": "ఎలా", "kn": "ಹೇಗೆ", "ml": "എങ്ങനെ"},

    # Negation
    "NOT": {"hi": "नहीं", "ta": "இல்லை", "te": "కాదు", "kn": "ಇಲ್ಲ", "ml": "ഇല്ല"},
    "NO": {"hi": "नहीं", "ta": "இல்லை", "te": "కాదు", "kn": "ಇಲ್ಲ", "ml": "ഇല്ല"},
}


class IndicTransformer:
    """Transforms ISL gloss sequences into fluent Hindi, Tamil, Telugu, Kannada, and Malayalam."""

    SUPPORTED_LANGUAGES = ("hi", "ta", "te", "kn", "ml")

    LETTER_PREFIX = {
        "hi": "अक्षर",
        "ta": "எழுத்து",
        "te": "అక్షరం",
        "kn": "ಅಕ್ಷರ",
        "ml": "അക്ഷരം",
    }

    @staticmethod
    def _collapse_char_runs(s: str) -> str:
        import re
        return re.sub(r'(.)\1+', r'\1', s).strip()

    def translate(self, tokens: Sequence[str], lang: str = "hi") -> str:
        """Translates an ISL gloss token sequence into the specified Indic language."""
        if not tokens:
            return ""

        clean_tokens = [str(t).strip().upper().replace("_", " ") for t in tokens if str(t).strip()]
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

        # 2. Fingerspelling Run Collapsing
        # e.g., ['C', 'A', 'T'], ['CC', 'AA', 'T'], ['CC', 'AA', 'TX'], or 'CCAAT'
        joined_chars = "".join(clean_tokens).replace(" ", "")
        collapsed_word = self._collapse_char_runs(joined_chars)

        if collapsed_word in CANONICAL_INDIC_IDIOMS:
            return CANONICAL_INDIC_IDIOMS[collapsed_word][norm_lang]

        if collapsed_word in INDIC_LEXICON:
            stem = INDIC_LEXICON[collapsed_word][norm_lang]
            return f"{stem}।" if norm_lang == "hi" else f"{stem}."

        # Handle 1-char noise suffix on fingerspelled words (e.g., 'CATX' -> 'CAT')
        if len(collapsed_word) > 2 and collapsed_word[:-1] in CANONICAL_INDIC_IDIOMS:
            return CANONICAL_INDIC_IDIOMS[collapsed_word[:-1]][norm_lang]

        # 3. Single Alphabet Letter
        if len(clean_tokens) == 1 and len(clean_tokens[0]) == 1 and clean_tokens[0].isalpha():
            letter = clean_tokens[0]
            prefix = self.LETTER_PREFIX.get(norm_lang, "ಅಕ್ಷರ")
            return f"{prefix} {letter}।" if norm_lang == "hi" else f"{prefix} {letter}."

        # 4. Compositional SOV Word Translation
        # In ISL and Indic languages, word order is naturally Subject -> Object -> Verb
        translated_words: List[str] = []
        is_question = any(t in ("WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO", "WHICH") for t in clean_tokens)
        has_negation = any(t in ("NOT", "NO", "NEVER") for t in clean_tokens)

        for token in clean_tokens:
            if token in ("NOT", "NO") and has_negation:
                continue  # Scoped at end for Indic verbs

            # Individual token run-collapse (e.g. 'CC' -> 'C')
            tok_norm = self._collapse_char_runs(token) if len(token) > 1 else token

            if tok_norm in INDIC_LEXICON and norm_lang in INDIC_LEXICON[tok_norm]:
                translated_words.append(INDIC_LEXICON[tok_norm][norm_lang])
            elif tok_norm in CANONICAL_INDIC_IDIOMS and norm_lang in CANONICAL_INDIC_IDIOMS[tok_norm]:
                translated_words.append(CANONICAL_INDIC_IDIOMS[tok_norm][norm_lang].rstrip("।."))
            elif token in INDIC_LEXICON and norm_lang in INDIC_LEXICON[token]:
                translated_words.append(INDIC_LEXICON[token][norm_lang])
            elif token in CANONICAL_INDIC_IDIOMS and norm_lang in CANONICAL_INDIC_IDIOMS[token]:
                translated_words.append(CANONICAL_INDIC_IDIOMS[token][norm_lang].rstrip("।."))
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
