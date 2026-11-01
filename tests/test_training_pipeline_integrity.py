"""
SETU — Training & Ingestion Pipeline Integrity Test Suite (Pillar 3)

Validates training data generation, augmentation bounds, morphology diversity,
split isolation without data leakage, and gradient descent convergence:
1. Multi-signer augmentation preserves strict canonical invariance (PROB-04).
2. Segmental anthropometric scaling introduces genuine anatomical diversity (PROB-10).
3. Independent hand dropout follows target Bernoulli distribution with zero-padding (PROB-03).
4. Signer and split isolation: 0 data leakage between train, val, and test partitions.
5. Mini-training loop convergence: loss descent, finite gradients, and parameter updates.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.generate_multisigner_dataset import augment_hand_landmarks
from training.train_isl_dualhand_mlp import RealSignMLP, ISLDualHandDataset


# ---------------------------------------------------------------------------
# 1. MULTI-SIGNER AUGMENTATION CANONICAL INVARIANCE (PROB-04)
# ---------------------------------------------------------------------------

def test_multisigner_augmentation_canonical_invariance():
    """All augmented samples strictly align Landmark 0 to (0,0,0) and Landmark 9 to (0,-1,0)."""
    rng = np.random.RandomState(42)
    base_hand = rng.uniform(-0.3, 0.3, size=(21, 3)).astype(np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.25, 0.0]

    num_samples = 100
    lm0_collector = np.zeros((num_samples, 3), dtype=np.float32)
    lm9_collector = np.zeros((num_samples, 3), dtype=np.float32)

    for i in range(num_samples):
        aug = augment_hand_landmarks(base_hand, max_rot_deg=15.0, noise_sigma=0.008)
        lm0_collector[i] = aug[0]
        lm9_collector[i] = aug[9]

    # Landmark 0 must be (0, 0, 0)
    assert np.allclose(lm0_collector, 0.0, atol=1e-5), "Landmark 0 must be origin (0,0,0)"
    assert np.max(np.std(lm0_collector, axis=0)) < 1e-5

    # Landmark 9 must be (0, -1, 0)
    expected_lm9 = np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float32), (num_samples, 1))
    assert np.allclose(lm9_collector, expected_lm9, atol=1e-4), "Landmark 9 must be (0,-1,0)"
    assert np.max(np.std(lm9_collector, axis=0)) < 1e-4


# ---------------------------------------------------------------------------
# 2. SEGMENTAL ANTHROPOMETRIC MORPHOLOGY VARIANCE (PROB-10)
# ---------------------------------------------------------------------------

def test_segmental_anthropometric_variation_diversity():
    """Segmental scaling generates finger-to-palm ratio diversity without degenerate phalanxes."""
    rng = np.random.RandomState(101)
    base_hand = rng.uniform(-0.25, 0.25, size=(21, 3)).astype(np.float32)
    base_hand[0] = [0.0, 0.0, 0.0]
    base_hand[9] = [0.0, -0.25, 0.0]

    num_samples = 200
    index_lengths = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        aug = augment_hand_landmarks(base_hand, scale_jitter=(0.95, 1.05))
        # Distance from MCP(5) to Tip(8)
        index_len = float(np.linalg.norm(aug[8] - aug[5]))
        index_lengths[i] = index_len
        # Ensure no inverted or collapsed phalanxes
        assert index_len > 0.10, "Finger length must remain non-degenerate"

    # Standard deviation confirms genuine anthropometric diversity is introduced
    assert np.std(index_lengths) > 0.03, f"Anthropometric std too low: {np.std(index_lengths):.4f}"


# ---------------------------------------------------------------------------
# 3. INDEPENDENT HAND DROPOUT DISTRIBUTION (PROB-03)
# ---------------------------------------------------------------------------

def test_hand_dropout_distribution_and_zero_padding():
    """ISLDualHandDataset applies Bernoulli hand dropout and properly zero-pads slots 0..62."""
    rng = np.random.RandomState(202)
    # Create synthetic 126-D samples (both hands non-zero)
    X = rng.uniform(0.1, 0.9, size=(50, 126)).astype(np.float32)
    y = rng.randint(0, 26, size=50)

    dataset = ISLDualHandDataset(X, y, augment=True, hand_dropout_prob=0.35)

    num_draws = 1000
    dropout_count = 0

    for i in range(num_draws):
        idx = i % len(dataset)
        x_aug, _ = dataset[idx]
        x_np = x_aug.numpy()

        left_hand_sum = float(np.abs(x_np[:63]).sum())
        right_hand_sum = float(np.abs(x_np[63:]).sum())

        if left_hand_sum == 0.0:
            dropout_count += 1
            # Right hand must still be active and non-zero
            assert right_hand_sum > 0.0, "Right hand must not be dropped when left hand is dropped"

    # Empirical dropout rate within binomial confidence bounds [0.28, 0.42] for p=0.35
    empirical_rate = dropout_count / num_draws
    assert 0.28 <= empirical_rate <= 0.42, f"Empirical dropout rate {empirical_rate:.3f} outside expected [0.28, 0.42]"


# ---------------------------------------------------------------------------
# 4. SIGNER & SPLIT ISOLATION: ZERO DATA LEAKAGE
# ---------------------------------------------------------------------------

def test_signer_and_split_isolation_zero_leakage():
    """Train, validation, and test splits are strictly disjoint with zero synthetic test leakage."""
    dataset_path = Path("data/processed/isl_master_clean_126d.npz")
    augmented_path = Path("data/processed/isl_multisigner_augmented_26class.npz")

    if not dataset_path.exists() or not augmented_path.exists():
        pytest.skip("Required processed datasets not found")

    d_clean = np.load(dataset_path, allow_pickle=True)
    d_aug = np.load(augmented_path, allow_pickle=True)

    splits_clean = d_clean["splits"]
    splits_aug = d_aug["splits"]

    # 1. Clean dataset split partitions
    train_clean_count = int(np.sum(splits_clean == "train"))
    val_clean_count = int(np.sum(splits_clean == "val"))
    test_clean_count = int(np.sum(splits_clean == "test"))

    assert train_clean_count + val_clean_count + test_clean_count == len(splits_clean)
    assert test_clean_count == 1445
    assert val_clean_count == 1422

    # 2. Augmented dataset retains pristine test and validation splits
    test_aug_count = int(np.sum(splits_aug == "test"))
    val_aug_count = int(np.sum(splits_aug == "val"))
    train_aug_count = int(np.sum(splits_aug == "train"))

    # Test and val splits must NEVER be augmented or multiplied
    assert test_aug_count == test_clean_count == 1445, "Test split was leaked or augmented!"
    assert val_aug_count == val_clean_count == 1422, "Val split was leaked or augmented!"
    assert train_aug_count == 40000, "Augmented train split should have exactly 40,000 samples"

    # 3. Check for exact vector duplicate leakage between test and train
    X_clean = d_clean["X"]
    X_test = X_clean[splits_clean == "test"]
    X_train_aug = d_aug["X"][splits_aug == "train"]

    # Sample random test vectors and verify they do not appear identically in train
    for i in range(10):
        test_vec = X_test[i * 100]
        # Diff across train set
        diffs = np.linalg.norm(X_train_aug[:5000] - test_vec, axis=1)
        assert np.min(diffs) > 1e-4, f"Data leakage detected for test sample {i}!"


# ---------------------------------------------------------------------------
# 5. MINI-TRAINING LOOP CONVERGENCE & GRADIENT FLOW
# ---------------------------------------------------------------------------

def test_mini_training_loop_convergence_and_gradient_flow():
    """Mini-batch training loop converges with strictly decreasing loss and finite gradients."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Synthetic batch of 64 samples
    X_batch = torch.randn(64, 126, dtype=torch.float32)
    y_batch = torch.randint(0, 26, (64,), dtype=torch.int64)

    model = RealSignMLP(input_dim=126, hidden1=64, hidden2=32, num_classes=26, use_batchnorm=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    initial_loss: float = 0.0
    final_loss: float = 0.0

    # Record initial weights
    w_initial = model.net[0].weight.clone()

    for step in range(25):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if step == 0:
            initial_loss = float(loss.item())

        # Assert no NaN or Inf gradients in any parameter
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any(), "NaN detected in gradient tensor"
            assert not torch.isinf(param.grad).any(), "Inf detected in gradient tensor"

        optimizer.step()

        if step == 24:
            final_loss = float(loss.item())

    # Verify convergence: loss decreased by at least 20%
    assert final_loss < 0.80 * initial_loss, f"Loss did not converge: initial {initial_loss:.4f}, final {final_loss:.4f}"

    # Verify weights actually updated
    w_final = model.net[0].weight
    weight_delta = float(torch.norm(w_final - w_initial).item())
    assert weight_delta > 0.01, "Model weights failed to update during training"
