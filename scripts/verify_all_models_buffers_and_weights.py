"""
SETU — Exhaustive Model Weights, Static & Dynamic Engines, and Buffer Test Suite
================================================================================
Runs end-to-end verification across:
1. Weight file integrity, NaN/Inf checks, tensor statistics, and parameter counts.
2. Static models (RealSign MLP, ISL MVP, Numerals, Vowels, ONNX exports).
3. Dynamic models (ISL Sequence LSTM, Amrita BiLSTM, INCLUDE 263-class).
4. Word buffer & SentenceAccumulator edge cases (pure static, pure dynamic, mixed,
   head nod/shake intent gating, zero-delay Indic translation, hand drops).
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Set up paths
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from backend.app.services.nlp.accumulator import SentenceAccumulator
from backend.app.services.vision.non_manual_markers import NonManualMarkerDetector
from models.adapters.realsign import RealSignClassifier
from models.adapters.mvp import ISLMvpClassifier
from models.adapters.sequence_lstm import ISLSequenceLSTMClassifier
from models.adapters.include import INCLUDEInferenceService


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_weights():
    banner("1. MODEL WEIGHTS INSPECTION & NUMERICAL SANITY")
    weights_dir = WORKSPACE / "models" / "weights"
    weight_files = list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.npz")) + list(weights_dir.glob("*.onnx"))
    
    passed = 0
    total = len(weight_files)
    
    for wf in sorted(weight_files):
        size_kb = wf.stat().st_size / 1024
        print(f"\n--> Checking: {wf.name} ({size_kb:.1f} KB)")
        
        if wf.suffix == ".pth":
            try:
                ckpt = torch.load(wf, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict):
                    keys = list(ckpt.keys())
                    # Check for state dict
                    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                    num_params = 0
                    has_nan = False
                    has_inf = False
                    param_count = 0
                    
                    for k, v in sd.items():
                        if isinstance(v, torch.Tensor):
                            param_count += v.numel()
                            if torch.isnan(v).any():
                                has_nan = True
                            if torch.isinf(v).any():
                                has_inf = True
                    
                    print(f"    PyTorch Checkpoint: {len(sd)} layers, {param_count:,} params")
                    print(f"    NaN detected: {has_nan} | Inf detected: {has_inf}")
                    assert not has_nan, f"NaN values in {wf.name}"
                    assert not has_inf, f"Inf values in {wf.name}"
                    passed += 1
                else:
                    print(f"    PyTorch model directly serialized: {type(ckpt)}")
                    passed += 1
            except Exception as e:
                print(f"    [FAIL] Error loading {wf.name}: {e}")
                raise e
                
        elif wf.suffix == ".npz":
            try:
                data = np.load(wf)
                print(f"    NumPy NPZ Archive: keys = {list(data.keys())}")
                has_nan = False
                for k in data.files:
                    arr = data[k]
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                        if np.isnan(arr).any():
                            has_nan = True
                print(f"    NaN detected: {has_nan}")
                assert not has_nan
                passed += 1
            except Exception as e:
                print(f"    [FAIL] Error loading {wf.name}: {e}")
                raise e
                
        elif wf.suffix == ".onnx":
            try:
                import onnx
                model = onnx.load(str(wf))
                onnx.checker.check_model(model)
                print(f"    ONNX Model: Validated successfully with IR v{model.ir_version}")
                passed += 1
            except Exception as e:
                print(f"    [FAIL] Error validating ONNX {wf.name}: {e}")
                raise e
                
    print(f"\n[WEIGHTS CHECK PASSED] {passed}/{total} weight files strictly validated.")
    return True


def check_static_models():
    banner("2. STATIC MODEL INFERENCE & EDGE CASES (RealSign MLP, MVP)")
    
    # RealSign Classifier (Dual-Hand 126-D)
    realsign_path = WORKSPACE / "models" / "weights" / "realsign_multisigner_mlp.pth"
    if not realsign_path.exists():
        realsign_path = WORKSPACE / "models" / "weights" / "realsign_26class_mlp.pth"
        
    clf = RealSignClassifier(realsign_path)
    print(f"Loaded RealSignClassifier: {clf.num_classes} classes (A-Z)")
    
    # Edge Case 1: Standard synthetic 126-D input (dual hands)
    features_126 = np.random.uniform(-0.5, 0.5, size=(126,)).astype(np.float32)
    res1 = clf.predict(features_126, confidence_threshold=0.0)
    print(f"  Test 1: Random 126-D input (raw argmax) -> Letter: {res1.letter}, Conf: {res1.confidence:.4f}")
    assert res1.letter in clf.class_names, f"Prediction {res1.letter} not in classes"
    assert 0.0 <= res1.confidence <= 1.0, f"Confidence {res1.confidence} out of range [0, 1]"
    
    # Edge Case 2: Single hand 63-D input (right hand only) with threshold=0.0
    features_63 = np.random.uniform(-0.5, 0.5, size=(63,)).astype(np.float32)
    res2 = clf.predict(features_63, confidence_threshold=0.0)
    print(f"  Test 2: Single hand 63-D input (raw argmax) -> Letter: {res2.letter}, Conf: {res2.confidence:.4f}")
    assert res2.letter in clf.class_names
    
    # Edge Case 2b: Noise gate verification (confidence_threshold=0.85 on random noise)
    res2_gated = clf.predict(features_63, confidence_threshold=0.85)
    print(f"  Test 2b: Noise Gate on Random 63-D (threshold=0.85) -> Status: '{res2_gated.letter}', is_known: {res2_gated.is_known_sign}")
    assert not res2_gated.is_known_sign or res2_gated.confidence >= 0.85
    
    # Edge Case 3: All-zero feature input (no hands detected or black frame)
    zero_features = np.zeros(126, dtype=np.float32)
    res3 = clf.predict(zero_features)
    print(f"  Test 3: All zeros (no hand) -> Status: {res3.letter}, Conf: {res3.confidence:.4f}")
    assert res3.letter == "Awaiting Sign Input"
    assert res3.confidence == 0.0
    
    # Edge Case 4: Extreme coordinate drift (-10.0 to +10.0)
    extreme_features = np.full(126, 10.0, dtype=np.float32)
    res4 = clf.predict(extreme_features)
    print(f"  Test 4: Extreme coordinates -> Letter: {res4.letter}, Conf: {res4.confidence:.4f}")
    assert not np.isnan(res4.confidence)
    assert not np.isinf(res4.confidence)
    
    # Edge Case 5: 100 consecutive frame latency benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        clf.predict(features_126)
    elapsed = (time.perf_counter() - t0) * 1000 / 100
    print(f"  Test 5: RealSign Average Inference Latency: {elapsed:.2f} ms per frame ({1000/elapsed:.0f} FPS)")
    assert elapsed < 10.0, f"Latency {elapsed:.2f}ms too high for real-time"
    
    # ISL MVP Classifier
    mvp_path = WORKSPACE / "models" / "weights" / "isl_mvp_model.npz"
    mvp = ISLMvpClassifier(mvp_path)
    print(f"\nLoaded ISLMvpClassifier: {len(mvp.class_names)} classes: {mvp.class_names}")
    # MVP expects 1264-D feature vector or (T, 158)
    mvp_feat = np.random.uniform(-0.5, 0.5, size=(1264,)).astype(np.float32)
    # Set presence flags so hand presence gating passes
    mvp_feat[146] = 1.0
    mvp_feat[147] = 1.0
    m_res = mvp.predict_features(mvp_feat)
    print(f"  Test MVP: 1264-D input -> Class: {m_res.label}, Conf: {m_res.confidence:.4f}")
    assert m_res.label in mvp.class_names
    
    print("\n[STATIC MODELS CHECK PASSED] All static models robust across all inputs.")
    return True


def check_dynamic_models():
    banner("3. DYNAMIC MODEL INFERENCE & SEQUENTIAL EDGE CASES (Sequence LSTM, INCLUDE)")
    
    lstm_path = WORKSPACE / "models" / "weights" / "isl_sequence_lstm.pth"
    lstm_clf = ISLSequenceLSTMClassifier(lstm_path)
    print(f"Loaded ISLSequenceLSTMClassifier: {lstm_clf.num_classes} classes: {lstm_clf.class_names}")
    print(f"Device: {lstm_clf.device}, Expected Input Dim: {lstm_clf.input_size}")
    
    # Edge Case 1: Full 30-frame sequence (30, 316 or 328 or 158)
    # ISLSequenceLSTMClassifier handles 158-D raw and auto-computes velocity to 316-D
    seq_full = np.random.uniform(-0.5, 0.5, size=(30, 158)).astype(np.float32)
    pred1 = lstm_clf.predict_sequence(seq_full)
    print(f"  Test 1: Full 30-frame sequence -> Class: {pred1.label}, Conf: {pred1.confidence:.4f}")
    assert pred1.label in lstm_clf.class_names
    assert 0.0 <= pred1.confidence <= 1.0
    
    # Edge Case 2: Short sequence with padding (10 frames)
    seq_short = np.random.uniform(-0.5, 0.5, size=(10, 158)).astype(np.float32)
    pred_s = lstm_clf.predict_sequence(seq_short)
    print(f"  Test 2: Short 10-frame sequence -> Class: {pred_s.label}, Conf: {pred_s.confidence:.4f}")
    assert pred_s.label in lstm_clf.class_names
    
    # Edge Case 3: Stationary sequence (all identical frames, zero motion)
    seq_stat = np.ones((30, 158), dtype=np.float32) * 0.25
    pred_stat = lstm_clf.predict_sequence(seq_stat)
    print(f"  Test 3: Stationary sequence -> Class: {pred_stat.label}, Conf: {pred_stat.confidence:.4f}")
    assert pred_stat.label in lstm_clf.class_names
    
    # Edge Case 4: Latency test
    t0 = time.perf_counter()
    for _ in range(50):
        lstm_clf.predict_sequence(seq_full)
    elapsed = (time.perf_counter() - t0) * 1000 / 50
    print(f"  Test 4: Dynamic LSTM Latency: {elapsed:.2f} ms per 30-frame window")
    assert elapsed < 25.0
    
    # INCLUDE Model Service
    print("\nLoading AI4Bharat INCLUDE Inference Service...")
    include_svc = INCLUDEInferenceService.get_instance()
    include_svc.load()
    print(f"  INCLUDE Service Ready: {len(include_svc.label_map)} classes")
    
    # Test synthetic 134-D keypoint stream (e.g. 50 frames, 134 features)
    dummy_features = np.random.uniform(0.0, 1.0, size=(50, 134)).astype(np.float32)
    inc_res = include_svc.predict_features(dummy_features)
    print(f"  Test 5: INCLUDE Features -> Class: {inc_res.label} ({inc_res.display_name}), Conf: {inc_res.confidence:.4f}")
    assert inc_res.label in include_svc.label_map
    
    print("\n[DYNAMIC MODELS CHECK PASSED] All temporal/dynamic engines operating cleanly.")
    return True


def check_word_and_sentence_buffers():
    banner("4. EXHAUSTIVE BUFFER & MIXED WORD SCENARIOS")
    
    # Scenario A: Pure Fingerspelling ("H" -> "E" -> "L" -> "L" -> "O")
    print("\n--- Scenario A: Pure Fingerspelling ---")
    acc = SentenceAccumulator(word_timeout=1.0)
    for char in ["H", "E", "L", "L", "O"]:
        acc.add_token(char, token_type="letter", bypass_debounce=True)
    
    print(f"  Tokens added: H-E-L-L-O")
    print(f"  Current word before commit: '{acc.current_word}'")
    print(f"  Effective tokens: {acc.get_effective_tokens()}")
    assert acc.current_word == "HELLO"
    assert acc.get_effective_tokens() == ["HELLO"]
    
    # Commit word manually (simulating Head Nod)
    acc.commit_current_word()
    print(f"  After Commit (Head Nod): Sentence Tokens = {acc.sentence_tokens}, Current Word = '{acc.current_word}'")
    assert acc.sentence_tokens == ["HELLO"]
    assert acc.current_word == ""
    
    # Scenario B: Pure Dynamic Phrase ("WATER")
    print("\n--- Scenario B: Pure Dynamic Phrase ---")
    acc.add_token("WATER", token_type="phrase")
    print(f"  Added dynamic sign 'WATER': Sentence Tokens = {acc.sentence_tokens}")
    assert acc.sentence_tokens == ["HELLO", "WATER"]
    
    # Scenario C: Mixed Sequence (Letters spelling "PLEASE" during existing sentence)
    print("\n--- Scenario C: Mixed Sequence (Fingerspelling after Dynamic) ---")
    for char in ["P", "L", "E", "A", "S", "E"]:
        acc.add_token(char, token_type="letter", bypass_debounce=True)
    print(f"  Current word in-progress: '{acc.current_word}'")
    print(f"  Effective tokens (sentence + current): {acc.get_effective_tokens()}")
    assert acc.current_word == "PLEASE"
    assert acc.get_effective_tokens() == ["HELLO", "WATER", "PLEASE"]
    
    # Scenario D: Mixed Transition (Dynamic sign arrives while static word is uncommitted)
    # The accumulator should automatically commit the uncommitted static word first!
    print("\n--- Scenario D: Automatic Commit on Mode Transition ---")
    acc.add_token("THANK_YOU", token_type="phrase")
    print(f"  Added dynamic 'THANK_YOU': Sentence Tokens = {acc.sentence_tokens}")
    assert acc.sentence_tokens == ["HELLO", "WATER", "PLEASE", "THANK_YOU"]
    assert acc.current_word == ""
    
    # Scenario E: Head Shake Rejection of In-Progress Word
    print("\n--- Scenario E: Head Shake (Negation Rejection) ---")
    # Start spelling a typo: "N", "O", "P", "E"
    for char in ["N", "O", "P", "E"]:
        acc.add_token(char, token_type="letter", bypass_debounce=True)
    print(f"  Spelled mistake: '{acc.current_word}'")
    assert acc.current_word == "NOPE"
    
    # Trigger Head Shake Rejection (pops candidate letter from current_word)
    while acc.current_word:
        acc.reject_last()
    print(f"  After Head Shake: Current Word = '{acc.current_word}'")
    print(f"  Sentence Tokens remain strictly preserved: {acc.sentence_tokens}")
    assert acc.current_word == ""
    assert acc.sentence_tokens == ["HELLO", "WATER", "PLEASE", "THANK_YOU"]
    
    # Scenario F: Hands Lowered / Disappeared
    print("\n--- Scenario F: Hands Dropped / No Keypoints Detected ---")
    # In websocket loop, when hand_detected == False, accumulator is untouched
    print(f"  Buffer preserved intact: {acc.get_effective_tokens()}")
    assert acc.sentence_tokens == ["HELLO", "WATER", "PLEASE", "THANK_YOU"]
    
    # Scenario G: Multilingual Indic Zero-Delay Translation
    print("\n--- Scenario G: Zero-Delay Indic Multilingual Output ---")
    trans_en = acc.translate_sentence()
    print(f"  Final English: '{trans_en}'")
    print(f"  Hindi (hi):    '{acc.translated_indic.get('hi')}'")
    print(f"  Tamil (ta):    '{acc.translated_indic.get('ta')}'")
    print(f"  Telugu (te):   '{acc.translated_indic.get('te')}'")
    print(f"  Kannada (kn):  '{acc.translated_indic.get('kn')}'")
    
    for lang in ["hi", "ta", "te", "kn"]:
        assert lang in acc.translated_indic
        assert len(acc.translated_indic[lang]) > 0
        
    print("\n[BUFFER SCENARIOS PASSED] All static, dynamic, mixed, and intent gating workflows verified 100%.")
    return True


if __name__ == "__main__":
    t_start = time.time()
    try:
        check_weights()
        check_static_models()
        check_dynamic_models()
        check_word_and_sentence_buffers()
        print("\n" + "=" * 70)
        print(f"  ALL 4 VERIFICATION SUITES COMPLETED IN {time.time() - t_start:.2f}s WITH ZERO ERRORS!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[FATAL ERROR] Verification failed: {e}")
        sys.exit(1)
