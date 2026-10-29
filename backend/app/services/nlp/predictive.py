"""
SETU — Modern Neural & Statistical Predictive Text Engine
Provides contextual next-word and phrase completion for Indian Sign Language (ISL)
conversations using Qwen2.5-0.5B on-device generative predictions with an LRU
in-memory cache and non-blocking background inference, backed by an expanded
statistical ISL conversational n-gram vocabulary.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("setu.services.predictive")

# Common conversational n-gram transitions in ISL domain
CONVERSATIONAL_BIGRAMS: Dict[str, List[str]] = {
    # Greetings & Courtesy
    "hello": ["how are you", "good morning", "my name is", "nice to meet you"],
    "good": ["morning", "afternoon", "evening", "night", "job"],
    "good morning": ["how are you", "teacher", "have a nice day", "everyone"],
    "good evening": ["how are you", "everyone", "welcome"],
    "good night": ["see you tomorrow", "sleep well", "take care"],
    "thank": ["you", "you very much", "you so much"],
    "thank you": ["very much", "for your help", "have a good day", "friend"],
    "namaste": ["everyone", "welcome", "how are you", "teacher"],
    "please": ["help me", "give me water", "repeat", "wait", "come here"],
    
    # Needs & Actions
    "i": ["need water", "am hungry", "want to learn", "am fine", "need help", "am ready"],
    "me": ["water please", "help needed", "give food", "listen to me"],
    "you": ["welcome", "need help", "are kind", "understand me", "where going"],
    "water": ["please", "bottle is empty", "need to drink", "cold water"],
    "food": ["hungry", "eat lunch", "want food", "delicious"],
    "eat": ["food", "lunch", "dinner", "breakfast"],
    "drink": ["water", "tea", "coffee", "juice"],
    
    # Emergency & Healthcare
    "help": ["me please", "needed urgently", "call doctor", "emergency"],
    "emergency": ["call doctor", "need hospital", "help urgently", "ambulance"],
    "doctor": ["hospital needed", "call doctor", "medicine time", "please help"],
    "medicine": ["take pills", "buy medicine", "headache", "fever"],
    "hospital": ["emergency", "doctor appointment", "need ride", "near here"],
    
    # Daily Life & Education
    "school": ["go to school", "bus coming", "teacher", "class time", "study"],
    "book": ["open book", "read book", "give book", "library"],
    "time": ["what is the time", "check clock", "time to go", "late"],
    "watch": ["what is the time", "check time", "wear watch", "my watch"],
    "phone": ["make a call", "hand me phone", "send message", "charge phone"],
    "mother": ["at home", "call mother", "cooking", "love you"],
    "father": ["at work", "call father", "coming home", "help me"],
    
    # Questions
    "what": ["is your name", "time is it", "happened", "do you want", "is this"],
    "where": ["are you going", "is bathroom", "is doctor", "is hospital"],
    "how": ["are you", "much does it cost", "to do this", "many people"],
    "why": ["did that happen", "are you sad", "are you late"],
    "when": ["are we leaving", "is class starting", "will doctor come"],
}

# Common words vocabulary for prefix completion
VOCABULARY: List[str] = [
    "hello", "thank you", "good morning", "good evening", "good night",
    "namaste", "welcome", "please", "help", "water", "food", "eat",
    "drink", "hungry", "thirsty", "bathroom", "medicine", "doctor",
    "hospital", "emergency", "happy", "sad", "tired", "fine",
    "understand", "repeat", "slowly", "yes", "no", "how", "what",
    "when", "where", "who", "why", "friend", "teacher", "student",
    "name", "sign language", "india", "setu", "learn", "study",
    "school", "book", "watch", "time", "phone", "mother", "father",
    "family", "home", "money", "buy", "give", "take", "stop", "go",
]


class PredictiveTextEngine:
    """
    Modern hybrid Neural + Statistical Predictive Text Engine.
    Leverages local Qwen2.5-0.5B causal completions with non-blocking caching
    and zero-latency statistical n-gram fallbacks.
    """

    def __init__(self, neural_translator: Optional[Any] = None) -> None:
        self.bigrams = CONVERSATIONAL_BIGRAMS
        self.vocab = sorted(set(VOCABULARY))
        self._neural_translator = neural_translator
        self._cache: Dict[str, List[str]] = {}
        self._pending_keys: Set[str] = set()
        self._lock = threading.Lock()

    def _get_neural_translator(self) -> Optional[Any]:
        if self._neural_translator is not None:
            return self._neural_translator
        try:
            from backend.app.services.nlp.neural_translator import neural_translator
            return neural_translator
        except Exception:
            return None

    def _neural_predict(self, text: str, limit: int = 3) -> List[str]:
        """Runs prompt-based causal completion on local Qwen2.5-0.5B."""
        translator = self._get_neural_translator()
        if translator is None or not getattr(translator, "is_loaded", False):
            return []

        try:
            import torch
            model = translator.model
            tokenizer = translator.tokenizer
            if model is None or tokenizer is None:
                return []

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a fast assistive text completion engine for sign language users. "
                        "Given the current sign language words or partial sentence, predict 3 distinct, "
                        "short, natural conversational continuations (1 to 4 words each). "
                        "Output ONLY the 3 suggestions separated by a vertical bar (|), no numbering or quotes."
                    ),
                },
                {"role": "user", "content": "Input: HELLO"},
                {"role": "assistant", "content": "how are you | good morning | nice to meet you"},
                {"role": "user", "content": "Input: WATER"},
                {"role": "assistant", "content": "please | drink cold water | bottle is empty"},
                {"role": "user", "content": "Input: WATCH"},
                {"role": "assistant", "content": "what is the time | check the time | where is it"},
                {"role": "user", "content": f"Input: {text}"},
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=24,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen = outputs[0][len(inputs.input_ids[0]):]
            raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
            suggestions = [s.strip().strip("\"'").lower() for s in raw.split("|") if s.strip()]
            return suggestions[:limit]
        except Exception as e:
            logger.debug("Neural predictive generation error: %s", e)
            return []

    def _bg_worker(self, key: str, limit: int) -> None:
        """Background thread worker to compute neural suggestions without blocking WebSocket loop."""
        try:
            results = self._neural_predict(key, limit=limit)
            if results:
                with self._lock:
                    self._cache[key] = results
        finally:
            with self._lock:
                self._pending_keys.discard(key)

    def predict_next(
        self,
        current_text: str,
        limit: int = 3,
        sync_neural: bool = False,
    ) -> List[str]:
        """
        Suggests the most probable next words or phrases based on the current sentence or last word.
        
        Args:
            current_text: Currently accumulated words or sentence.
            limit: Maximum number of suggestions to return.
            sync_neural: When True, awaits neural prediction synchronously (useful for tests/APIs).
            
        Returns:
            List of suggested completions.
        """
        if not current_text or not current_text.strip():
            return ["hello", "namaste", "good morning"][:limit]

        clean_text = current_text.strip().lower()

        # 1. Check in-memory cache
        with self._lock:
            if clean_text in self._cache and self._cache[clean_text]:
                return self._cache[clean_text][:limit]

        # 2. Synchronous neural computation if explicitly requested
        if sync_neural:
            neural_res = self._neural_predict(clean_text, limit=limit)
            if neural_res:
                with self._lock:
                    self._cache[clean_text] = neural_res
                return neural_res[:limit]

        # 3. Trigger non-blocking background neural worker if translator is loaded
        translator = self._get_neural_translator()
        if translator is not None and getattr(translator, "is_loaded", False):
            with self._lock:
                if clean_text not in self._pending_keys:
                    self._pending_keys.add(clean_text)
                    threading.Thread(
                        target=self._bg_worker,
                        args=(clean_text, limit),
                        daemon=True,
                        name=f"PredWorker-{clean_text[:10]}",
                    ).start()

        # 4. Instant zero-latency statistical fallback (returned while neural worker runs)
        tokens = clean_text.split()
        if not tokens:
            return ["hello", "namaste", "good morning"][:limit]

        # Check bigram for last 2 tokens
        if len(tokens) >= 2:
            two_word_key = f"{tokens[-2]} {tokens[-1]}"
            if two_word_key in self.bigrams:
                return self.bigrams[two_word_key][:limit]

        # Check bigram for last token
        last_token = tokens[-1]
        if last_token in self.bigrams:
            return self.bigrams[last_token][:limit]

        # Check prefix matching if the user is typing/fingerspelling a partial word
        prefix_matches = [
            w for w in self.vocab if w.startswith(last_token) and w != last_token
        ]
        if prefix_matches:
            return prefix_matches[:limit]

        # Context-aware fallback defaults
        if any(w in clean_text for w in ("doctor", "hospital", "pain", "sick", "emergency")):
            return ["call doctor", "need medicine", "help urgently"][:limit]
        if any(w in clean_text for w in ("water", "food", "eat", "drink", "hungry")):
            return ["please", "need water", "want food"][:limit]
        if any(w in clean_text for w in ("time", "watch", "clock")):
            return ["what is the time", "check time", "late"][:limit]

        return ["thank you", "please", "help"][:limit]
