"""
SETU — Specialized Domain Vocabulary Packs (PROB-24) & Number/Currency Engine (PROB-18).

Provides clinical healthcare and banking dictionaries, priority predictive suggestions,
and automated grouping of sequential numerical digits and Indian currency signs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class DomainWord:
    token: str
    display_name: str
    category: str
    translations: Dict[str, str]  # en, hi, ta, te, kn


HEALTHCARE_PACK: Dict[str, DomainWord] = {
    "PAIN": DomainWord("PAIN", "Pain", "symptom", {
        "en": "Pain", "hi": "दर्द", "ta": "வலி", "te": "నొప్పి", "kn": "ನೋವು"
    }),
    "CHEST PAIN": DomainWord("CHEST PAIN", "Chest Pain", "symptom", {
        "en": "Chest pain", "hi": "सीने में दर्द", "ta": "மார்பு வலி", "te": "ఛాతీ నొప్పి", "kn": "ಎದೆ ನೋವು"
    }),
    "FEVER": DomainWord("FEVER", "Fever", "symptom", {
        "en": "Fever", "hi": "बुखार", "ta": "காய்ச்சல்", "te": "జ్వరం", "kn": "ಜ್ವರ"
    }),
    "COUGH": DomainWord("COUGH", "Cough", "symptom", {
        "en": "Cough", "hi": "खाँसी", "ta": "இருமல்", "te": "దగ్గు", "kn": "ಕೆಮ್ಮು"
    }),
    "HEADACHE": DomainWord("HEADACHE", "Headache", "symptom", {
        "en": "Headache", "hi": "सिरदर्द", "ta": "தலைவலி", "te": "తలనొప్పి", "kn": "ತಲೆನೋವು"
    }),
    "MEDICINE": DomainWord("MEDICINE", "Medicine", "treatment", {
        "en": "Medicine", "hi": "दवाई", "ta": "மருந்து", "te": "మందు", "kn": "ಔಷಧಿ"
    }),
    "DOCTOR": DomainWord("DOCTOR", "Doctor", "personnel", {
        "en": "Doctor", "hi": "डॉक्टर", "ta": "மருத்துவர்", "te": "వైద్యుడు", "kn": "ವೈದ್ಯರು"
    }),
    "NURSE": DomainWord("NURSE", "Nurse", "personnel", {
        "en": "Nurse", "hi": "नर्स", "ta": "செவிலியர்", "te": "నర్సు", "kn": "ದಾದಿ"
    }),
    "HOSPITAL": DomainWord("HOSPITAL", "Hospital", "facility", {
        "en": "Hospital", "hi": "अस्पताल", "ta": "மருத்துவமனை", "te": "ఆసుపత్రి", "kn": "ಆಸ್ಪತ್ರೆ"
    }),
    "ALLERGY": DomainWord("ALLERGY", "Allergy", "condition", {
        "en": "Allergy", "hi": "एलर्जी", "ta": "ஒவ்வாமை", "te": "అలెర్జీ", "kn": "ಅಲರ್ಜಿ"
    }),
    "BLOOD PRESSURE": DomainWord("BLOOD PRESSURE", "Blood Pressure", "vitals", {
        "en": "Blood Pressure", "hi": "रक्तचाप (BP)", "ta": "இரத்த அழுத்தம்", "te": "రక్తపోటు", "kn": "ರಕ್ತದೊತ್ತಡ"
    }),
    "WHERE HURTS": DomainWord("WHERE HURTS", "Where does it hurt?", "question", {
        "en": "Where does it hurt?", "hi": "कहाँ दर्द हो रहा है?", "ta": "எங்கு வலிக்கிறது?", "te": "ఎక్కడ నొప్పిగా ఉంది?", "kn": "ಎಲ್ಲಿ ನೋವಾಗುತ್ತಿದೆ?"
    }),
}


BANKING_PACK: Dict[str, DomainWord] = {
    "BANK": DomainWord("BANK", "Bank", "institution", {
        "en": "Bank", "hi": "बैंक", "ta": "வங்கி", "te": "బ్యాంకు", "kn": "ಬ್ಯಾಂಕ್"
    }),
    "MONEY": DomainWord("MONEY", "Money", "finance", {
        "en": "Money", "hi": "पैसे / धन", "ta": "பணம்", "te": "డబ్బు", "kn": "ಹಣ"
    }),
    "ACCOUNT": DomainWord("ACCOUNT", "Account", "banking", {
        "en": "Account", "hi": "खाता", "ta": "கணக்கு", "te": "ఖాతా", "kn": "ಖಾತೆ"
    }),
    "ATM": DomainWord("ATM", "ATM", "facility", {
        "en": "ATM", "hi": "एटीएम", "ta": "ஏடிஎம்", "te": "ఎటిఎం", "kn": "ಎಟಿಎಂ"
    }),
    "CHEQUE": DomainWord("CHEQUE", "Cheque", "instrument", {
        "en": "Cheque", "hi": "चेक", "ta": "காசோலை", "te": "చెక్కు", "kn": "ಚೆಕ್"
    }),
    "SIGNATURE": DomainWord("SIGNATURE", "Signature", "verification", {
        "en": "Signature", "hi": "हस्ताक्षर", "ta": "கையொப்பம்", "te": "సంతకం", "kn": "ಸಹಿ"
    }),
    "CARD": DomainWord("CARD", "Card", "instrument", {
        "en": "Debit / Credit Card", "hi": "कार्ड", "ta": "அட்டை", "te": "కార్డు", "kn": "ಕಾರ್ಡ್"
    }),
    "DEPOSIT": DomainWord("DEPOSIT", "Deposit", "action", {
        "en": "Deposit", "hi": "जमा करना", "ta": "டெபாசிட்", "te": "డిపాజిట్", "kn": "ಠೇವಣಿ"
    }),
    "WITHDRAW": DomainWord("WITHDRAW", "Withdraw", "action", {
        "en": "Withdraw", "hi": "पैसे निकालना", "ta": "பணம் எடுப்பது", "te": "డబ్బు డ్రా చేయడం", "kn": "ಹಣ ಹಿಂಪಡೆಯುವುದು"
    }),
    "AADHAAR": DomainWord("AADHAAR", "Aadhaar Card", "identity", {
        "en": "Aadhaar Card", "hi": "आधार कार्ड", "ta": "ஆதார் அட்டை", "te": "ఆధార్ కార్డు", "kn": "ಆಧಾರ್ ಕಾರ್ಡ್"
    }),
    "BALANCE": DomainWord("BALANCE", "Balance", "inquiry", {
        "en": "Account Balance", "hi": "खाता शेष (बैलेंस)", "ta": "கணக்கு இருப்பு", "te": "ఖాతా నిల్వ", "kn": "ಖಾತೆ ಬಾಕಿ"
    }),
}


def format_tokens_with_numbers_and_currency(tokens: Sequence[str]) -> List[str]:
    """
    Groups contiguous single-digit tokens into numerical sequences,
    and applies Indian currency formatting:
    ['9', '8', '2', '0'] -> ['9820']
    ['RUPEES', '5', '0', '0'] -> ['₹500']
    """
    if not tokens:
        return []

    result: List[str] = []
    digit_buffer: List[str] = []
    currency_prefix: bool = False

    for t in tokens:
        clean = str(t).strip().upper()
        if clean in ("RUPEES", "RUPEE", "RS", "INR", "₹"):
            currency_prefix = True
            continue

        if clean.isdigit():
            if len(clean) == 1:
                digit_buffer.append(clean)
            else:
                if digit_buffer:
                    combined = "".join(digit_buffer) + clean
                    digit_buffer.clear()
                else:
                    combined = clean
                if currency_prefix:
                    result.append(f"₹{combined}")
                    currency_prefix = False
                else:
                    result.append(combined)
        else:
            # Flush accumulated digits
            if digit_buffer:
                num_str = "".join(digit_buffer)
                if currency_prefix:
                    result.append(f"₹{num_str}")
                    currency_prefix = False
                else:
                    result.append(num_str)
                digit_buffer.clear()

            if currency_prefix:
                result.append("₹")
                currency_prefix = False

            result.append(clean)

    # Flush any remaining digits
    if digit_buffer:
        num_str = "".join(digit_buffer)
        if currency_prefix:
            result.append(f"₹{num_str}")
        else:
            result.append(num_str)

    return result


class DomainPacksManager:
    """Manages context-aware clinical and financial vocabulary packs."""

    def __init__(self) -> None:
        self.healthcare = HEALTHCARE_PACK
        self.banking = BANKING_PACK
        self.active_domain: str = "general"

    def set_domain(self, domain: str) -> None:
        self.active_domain = domain.lower().strip()

    def get_domain_suggestions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns top suggested signs for the active domain context."""
        pack = self.healthcare if self.active_domain == "healthcare" else (
            self.banking if self.active_domain == "banking" else {}
        )
        return [
            {
                "token": word.token,
                "display_name": word.display_name,
                "translations": word.translations,
            }
            for word in list(pack.values())[:limit]
        ]


# Singleton instance
domain_packs_manager = DomainPacksManager()
