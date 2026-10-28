"""Unit tests for SETU Neural Gloss-to-English Translator & Accumulator Integration."""

from backend.app.services.nlp.neural_translator import NeuralGlossTranslator
from backend.app.services.nlp.accumulator import SentenceAccumulator


def test_neural_translator_fallback():
    translator = NeuralGlossTranslator(model_id="mock-model")
    assert translator.is_loaded is False

    # Empty input
    assert translator.translate([]) == ""
    assert translator.translate("") == ""

    # Heuristic fallback when model is not loaded
    res = translator.translate(["ME", "WATER", "DRINK"])
    assert "Me water drink." in res

    # String input with duplicate tokens collapsed
    res_str = translator.translate("HELLO HELLO GOOD MORNING")
    assert "Hello good morning." in res_str


def test_sentence_accumulator_with_mock_neural_translator():
    class MockNeuralTranslator:
        is_loaded = True

        def translate(self, tokens):
            return "The doctor will hurry to the hospital."

    mock_translator = MockNeuralTranslator()
    accumulator = SentenceAccumulator(neural_translator=mock_translator)

    accumulator.add_token("DOCTOR", token_type="phrase")
    accumulator.add_token("HOSPITAL", token_type="phrase")
    accumulator.add_token("HURRY", token_type="phrase")

    result = accumulator.translate_sentence()
    assert result == "The doctor will hurry to the hospital."
    assert accumulator.translated_text == "The doctor will hurry to the hospital."


def test_sentence_accumulator_fallback_to_rules():
    class OfflineTranslator:
        is_loaded = False

    offline_translator = OfflineTranslator()
    accumulator = SentenceAccumulator(neural_translator=offline_translator)

    # Standard ISL rule: ME WATER DRINK -> I want water
    accumulator.add_token("ME", token_type="phrase")
    accumulator.add_token("WATER", token_type="phrase")
    accumulator.add_token("DRINK", token_type="phrase")

    result = accumulator.translate_sentence()
    assert result == "I want water"

