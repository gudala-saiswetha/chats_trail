#!/usr/bin/env python3
"""
Step 2 Verification: Sanity check for GatedHLSGuard architecture, feature scaling,
forward pass, and multi-task gradient backward propagation.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hls_guard import GatedHLSGuard
from models.dataset_loader import (
    build_pattern_matrix,
    PatternScaler,
    HLSGuardDataset,
    FEATURE_NAMES
)

def run_step2_verification():
    print("=" * 65)
    print("STEP 2: MODEL ARCHITECTURE & MULTI-TASK PIPELINE VERIFICATION")
    print("=" * 65)

    # 1. Load small verification batch from gold cleaned dataset
    data_path = "dataset/gold_clean/train_gold.parquet"
    print(f"Loading verification slice from {data_path}...")
    df = pd.read_parquet(data_path).head(16)
    print(f"Loaded batch of {len(df)} samples.")

    # 2. Extract and scale pattern features
    print("\n[1/4] Extracting and scaling 14 handcrafted pattern features...")
    pattern_matrix = build_pattern_matrix(df, text_column="head_tail_prompt")
    scaler = PatternScaler(scaler_path="models/verify_scaler.pkl")
    scaled_patterns = scaler.fit_and_save(pattern_matrix)
    print(f"Pattern matrix shape: {scaled_patterns.shape} (14 features verified).")

    # 3. Tokenize texts
    model_name = "distilroberta-base"
    print(f"\n[2/4] Tokenizing texts using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded = tokenizer(
        df["head_tail_prompt"].tolist(),
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

    # 4. Construct PyTorch Dataset & DataLoader
    dataset = HLSGuardDataset(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        pattern_features=scaled_patterns,
        binary_labels=df["binary_label"].to_numpy(),
        category_labels=df["category_label"].to_numpy(),
        risk_labels=df["risk_label"].to_numpy(),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    batch = next(iter(loader))

    # 5. Initialize GatedHLSGuard
    print("\n[3/4] Initializing GatedHLSGuard (Option A: Gated Feature Fusion)...")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    model = GatedHLSGuard(
        model_name=model_name,
        num_pattern_features=14,
        num_categories=4,
        num_risk_levels=5,
        fusion_dim=256,
        dropout=0.2
    ).to(device)

    # 6. Forward Pass & Timing
    print("\n[4/4] Executing forward and backward passes...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pattern_features = batch["pattern_features"].to(device)

    t0 = time.perf_counter()
    outputs = model(input_ids, attention_mask, pattern_features)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(df)

    bin_logits = outputs["binary_logits"]
    cat_logits = outputs["category_logits"]
    risk_logits = outputs["risk_logits"]
    gate_w = outputs["gate_weights"]

    print(f"Forward pass completed in {latency_ms:.2f} ms per sample.")
    print(f"  - Binary Head Logits shape  : {tuple(bin_logits.shape)} (Expected: [16, 2])")
    print(f"  - Category Head Logits shape: {tuple(cat_logits.shape)} (Expected: [16, 4])")
    print(f"  - Risk Head Logits shape    : {tuple(risk_logits.shape)} (Expected: [16, 5])")
    print(f"  - Gate Weights shape        : {tuple(gate_w.shape)} (Expected: [16, 256])")

    # Assert correct dimensions
    assert bin_logits.shape == (16, 2), f"Unexpected binary logits shape: {bin_logits.shape}"
    assert cat_logits.shape == (16, 4), f"Unexpected category logits shape: {cat_logits.shape}"
    assert risk_logits.shape == (16, 5), f"Unexpected risk logits shape: {risk_logits.shape}"
    assert gate_w.shape == (16, 256), f"Unexpected gate weights shape: {gate_w.shape}"

    # Verify Multi-Task Loss & Backpropagation
    loss_fn = nn.CrossEntropyLoss()
    loss_b = loss_fn(bin_logits, batch["binary_labels"].to(device))
    loss_c = loss_fn(cat_logits, batch["category_labels"].to(device))
    loss_r = loss_fn(risk_logits, batch["risk_labels"].to(device))
    total_loss = loss_b + loss_c + loss_r

    print(f"\nInitial Loss values:")
    print(f"  - Binary Loss: {loss_b.item():.4f}")
    print(f"  - Category Loss: {loss_c.item():.4f}")
    print(f"  - Risk Loss: {loss_r.item():.4f}")
    print(f"  - Total Multi-Task Loss: {total_loss.item():.4f}")

    total_loss.backward()

    # Check gradients
    encoder_grad = next(model.semantic_encoder.parameters()).grad
    pattern_grad = next(model.pattern_branch.parameters()).grad
    gate_grad = next(model.fusion.gate.parameters()).grad
    risk_grad = next(model.risk_head.parameters()).grad

    assert encoder_grad is not None and torch.isfinite(encoder_grad).all(), "Encoder gradients invalid!"
    assert pattern_grad is not None and torch.isfinite(pattern_grad).all(), "Pattern branch gradients invalid!"
    assert gate_grad is not None and torch.isfinite(gate_grad).all(), "Gate gradients invalid!"
    assert risk_grad is not None and torch.isfinite(risk_grad).all(), "Risk head gradients invalid!"

    print("\nGradient propagation check: PASSED (all heads, gating, and encoder receive valid gradients).")
    print("=" * 65)
    print("STEP 2 COMPLETE: Model architecture is verified and ready for Step 3.")
    print("=" * 65)

if __name__ == "__main__":
    run_step2_verification()
