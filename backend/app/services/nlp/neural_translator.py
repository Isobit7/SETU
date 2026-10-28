"""
SETU — Neural ISL Gloss-to-English Sequence-to-Sequence Translator
Powered by lightweight Qwen2.5-0.5B-Instruct for open-vocabulary,
fluent English sentence generation from raw sign language tokens.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger("setu.services.neural_translator")

LOCAL_MODEL_DIR = Path("models/weights/qwen_0.5b")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class NeuralGlossTranslator:
    """
    Translates arbitrary sequences of ISL glosses, words, and fingerspelled letters
    into fluent, natural, grammatically correct English sentences using a compact
    local Sequence-to-Sequence / Small Language Model.
    """

    _instance: Optional["NeuralGlossTranslator"] = None
    _lock = threading.Lock()

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.is_loading = False
        self.device = "cpu"

    @classmethod
    def get_instance(cls, model_id: str = DEFAULT_MODEL_ID) -> "NeuralGlossTranslator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(model_id=model_id)
            return cls._instance

    def load(self, async_load: bool = False) -> None:
        """Loads tokenizer and neural model weights into memory (CUDA if available)."""
        if self.is_loaded or self.is_loading:
            return

        if async_load:
            threading.Thread(target=self._load_sync, daemon=True, name="QwenLoadThread").start()
        else:
            self._load_sync()

    def _load_sync(self) -> None:
        self.is_loading = True
        try:
            import os
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            has_cuda = torch.cuda.is_available()
            self.device = "cuda" if has_cuda else "cpu"
            load_target = str(LOCAL_MODEL_DIR.resolve()) if (LOCAL_MODEL_DIR / "model.safetensors").exists() else self.model_id
            logger.info("Loading NeuralGlossTranslator (%s) on %s...", load_target, self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(load_target)
            self.model = AutoModelForCausalLM.from_pretrained(
                load_target,
                torch_dtype=torch.float16 if has_cuda else torch.float32,
                device_map="auto" if has_cuda else None,
            )
            self.is_loaded = True
            logger.info("NeuralGlossTranslator loaded successfully on %s.", self.device)
        except Exception as e:
            logger.warning("Could not load NeuralGlossTranslator (%s): %s", self.model_id, e)
            self.is_loaded = False
        finally:
            self.is_loading = False

    def translate(self, tokens: Union[Sequence[str], str]) -> str:
        """
        Translates raw gloss tokens into a natural English sentence.
        
        Args:
            tokens: Sequence of words/letters or a single space-separated string.
            
        Returns:
            Grammatically fluent English sentence.
        """
        if isinstance(tokens, str):
            clean_str = tokens.strip()
        else:
            clean_str = " ".join(str(t).strip() for t in tokens if str(t).strip())

        if not clean_str:
            return ""

        # Normalize consecutive duplicate tokens
        words = clean_str.split()
        deduped = []
        for w in words:
            if not deduped or deduped[-1].upper() != w.upper():
                deduped.append(w)
        gloss_input = " ".join(deduped)

        if not self.is_loaded or self.model is None or self.tokenizer is None:
            # Fallback to Title-cased join if model is not loaded
            return gloss_input.capitalize() + "."

        try:
            import torch

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an Indian Sign Language (ISL) to fluent English translator. "
                        "Convert raw sign words into one natural, fluent, grammatically correct English sentence. "
                        "Output ONLY the English sentence without extra text or quotation marks."
                    ),
                },
                {"role": "user", "content": "ISL: ME WATER DRINK"},
                {"role": "assistant", "content": "I want to drink water."},
                {"role": "user", "content": "ISL: MOTHER MEDICINE BUY NEED"},
                {"role": "assistant", "content": "My mother needs to buy medicine."},
                {"role": "user", "content": "ISL: MY NAME S A T H W I K"},
                {"role": "assistant", "content": "My name is Sathwik."},
                {"role": "user", "content": f"ISL: {gloss_input}"},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=36,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Slice out generated response tokens past the input prompt
            gen_tokens = outputs[0][len(inputs.input_ids[0]):]
            result = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Clean output: strip surrounding quotes and trailing junk
            result = re.sub(r'^["\']|["\']$', "", result).strip()
            if result and not result.endswith((".", "!", "?")):
                result += "."

            return result
        except Exception as e:
            logger.warning("NeuralGlossTranslator inference error: %s. Using heuristic fallback.", e)
            return gloss_input.capitalize() + "."


# Global singleton instance
neural_translator = NeuralGlossTranslator.get_instance()

