"""
Unit tests for ISLSyntaxTransformer.
Verifies SOV to SVO translation, question fronting, negation scoping, and fingerspelling.
"""

from backend.app.services.nlp.syntax_transformer import ISLSyntaxTransformer


def test_wh_question_fronting():
    tx = ISLSyntaxTransformer()
    
    assert tx.transform(["YOU", "NAME", "WHAT"]) == "What is your name?"
    assert tx.transform(["NAME", "WHAT"]) == "What is the name?"
    assert tx.transform(["TIME", "WHAT"]) == "What time is it?"
    assert tx.transform(["YOU", "LIVE", "WHERE"]) == "Where do you live?"
    assert tx.transform(["YOU", "GO", "WHERE"]) == "Where are you going?"
    assert tx.transform(["HOSPITAL", "WHERE"]) == "Where is the hospital?"
    assert tx.transform(["TRAIN", "DELAY", "WHY"]) == "Why is it delayed?"
    assert tx.transform(["YOU", "HOW"]) == "How are you?"


def test_negation_scoping():
    tx = ISLSyntaxTransformer()
    
    # "I TEA LIKE NOT" -> "I do not like tea."
    res1 = tx.transform(["I", "TEA", "LIKE", "NOT"])
    assert res1 == "I do not like tea."
    
    # "HE COME NOT" -> "He does not come."
    res2 = tx.transform(["HE", "COME", "NOT"])
    assert res2 == "He does not come."
    
    # "NOT" alone -> "No."
    res3 = tx.transform(["NOT"])
    assert res3 == "No."


def test_sov_to_svo_reordering():
    tx = ISLSyntaxTransformer()
    
    # "ME WATER WANT" -> "I want some water."
    res1 = tx.transform(["ME", "WATER", "WANT"])
    assert res1 == "I want some water."
    
    # "HE FOOD EAT" -> "He eats some food."
    res2 = tx.transform(["HE", "FOOD", "EAT"])
    assert res2 == "He eats some food."
    
    # Single courtesy terms
    assert tx.transform(["WATER"]) == "I would like some water, please."
    assert tx.transform(["FOOD"]) == "I would like some food, please."
    assert tx.transform(["HELP"]) == "Please help me."
    assert tx.transform(["THANK_YOU"]) == "Thank you very much."


def test_fingerspelling_concatenation_and_spellcheck():
    tx = ISLSyntaxTransformer()
    
    # Clean spelled word
    assert tx.transform(["H", "E", "L", "P"]) == "Help"
    assert tx.transform(["W", "A", "T", "E", "R"]) == "Water"
    assert tx.transform(["D", "O", "C", "T", "O", "R"]) == "Doctor"


def test_empty_and_unknown_safety():
    tx = ISLSyntaxTransformer()
    
    assert tx.transform([]) == ""
    # Unknown tokens without verbs or subjects should return empty string so fallback operates
    assert tx.transform(["XYZ", "ABC"]) == ""

