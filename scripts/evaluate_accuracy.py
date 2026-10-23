#!/usr/bin/env python3
"""
HLS-Guard: Independent, Segregated Accuracy Evaluation Engine
Runs on both Local Mac and Google Colab with separate, isolated outputs.

Usage:
  Mac:
    python3 scripts/evaluate_accuracy.py --env mac \
      --checkpoint checkpoints_gold/best_hls_guard.pt \
      --scaler checkpoints_gold/pattern_scaler.pkl \
      --output_report benchmarks/gold_accuracy_report_mac.json

  Colab:
    python3 scripts/evaluate_accuracy.py --env colab \
      --checkpoint checkpoints_colab/best_hls_guard.pt \
      --scaler checkpoints_colab/pattern_scaler.pkl \
      --output_report benchmarks/gold_accuracy_report_colab.json
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hls_guard import GatedHLSGuard
from models.dataset_loader import build_pattern_matrix, PatternScaler

def main():
    parser = argparse.ArgumentParser(description="Evaluate HLS-Guard Accuracy (Segregated Mac / Colab)")
    parser.add_argument("--env", type=str, default="mac", choices=["mac", "colab"], help="Environment tag (mac or colab)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_gold/best_hls_guard.pt", help="Path to model checkpoint")
    parser.add_argument("--scaler", type=str, default="checkpoints_gold/pattern_scaler.pkl", help="Path to pattern scaler")
    parser.add_argument("--test_path", type=str, default="dataset/gold_clean/test_gold.parquet", help="Path to clean test dataset")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of test prompts to evaluate (0 = ALL 47,748 prompts)")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--output_report", type=str, default="", help="Path to save output JSON report")
    args = parser.parse_args()

    if not args.output_report:
        args.output_report = f"benchmarks/gold_accuracy_report_{args.env}.json"

    print("=" * 75)
    print(f"HLS-GUARD: ACCURACY EVALUATION ENGINE [ENVIRONMENT: {args.env.upper()}]")
    print("=" * 75)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    print(f"Compute Device  : {device}")
    print(f"Checkpoint Path : {args.checkpoint}")
    print(f"Scaler Path     : {args.scaler}")
    print(f"Test Dataset    : {args.test_path}")
    print(f"Report Output   : {args.output_report}")

    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] Checkpoint not found at: {args.checkpoint}")
        print("Please train the model first before running this test:")
        print(f"  python3 scripts/train_gold_model.py")
        sys.exit(1)

    # 1. Load Test Dataset
    print(f"\n[1/4] Loading test dataset...")
    test_df = pd.read_parquet(args.test_path)
    total_available = len(test_df)

    if args.sample_size and args.sample_size < total_available:
        print(f"  Sampling {args.sample_size:,} balanced test prompts for fast evaluation...")
        n_half = args.sample_size // 2
        mal = test_df[test_df["binary_label"] == 1].sample(n=n_half, random_state=42)
        ben = test_df[test_df["binary_label"] == 0].sample(n=n_half, random_state=42)
        test_df = pd.concat([mal, ben]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Evaluating on: {len(test_df):,} prompts ({int(test_df['binary_label'].sum()):,} Attacks, {len(test_df) - int(test_df['binary_label'].sum()):,} Safe)")

    # 2. Extract and Scale Pattern Features
    print("\n[2/4] Transforming 14 pattern features...")
    scaler = PatternScaler(scaler_path=args.scaler)
    test_patterns = scaler.load_and_transform(build_pattern_matrix(test_df, "head_tail_prompt"))

    # 3. Load Model
    print(f"\n[3/4] Loading model weights from: {args.checkpoint} ...")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = GatedHLSGuard("distilroberta-base", num_pattern_features=14, num_categories=4, num_risk_levels=5).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Run Test Inference
    print(f"\n[4/4] Executing batch inference across {len(test_df):,} test prompts...")
    y_true = test_df["binary_label"].to_numpy()
    probs = []
    preds = []

    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    all_test_texts = test_df["head_tail_prompt"].tolist()

    with torch.no_grad():
        for i in range(0, len(test_df), args.batch_size):
            batch_texts = all_test_texts[i:i + args.batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            pat = torch.tensor(test_patterns[i:i + args.batch_size], dtype=torch.float32).to(device)

            out = model(enc["input_ids"], enc["attention_mask"], pat)
            b_probs = torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy()
            b_preds = out["binary_logits"].argmax(dim=-1).cpu().numpy()

            probs.extend(b_probs)
            preds.extend(b_preds)

    probs = np.array(probs)
    preds = np.array(preds)

    # Calculate exact standard metrics
    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
    cm = confusion_matrix(y_true, preds)
    fpr = cm[0, 1] / max(cm[0, 0] + cm[0, 1], 1)
    roc_auc = roc_auc_score(y_true, probs)

    # Save detailed JSON report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report_data = {
        "environment": args.env,
        "checkpoint_evaluated": args.checkpoint,
        "test_dataset": args.test_path,
        "total_prompts_evaluated": len(test_df),
        "standard_binary_accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall_catch_rate": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),
        "roc_auc_score": round(float(roc_auc), 4),
        "confusion_matrix": {
            "true_negatives_safe_allowed": int(cm[0, 0]),
            "false_positives_safe_blocked": int(cm[0, 1]),
            "false_negatives_attack_missed": int(cm[1, 0]),
            "true_positives_attack_blocked": int(cm[1, 1])
        }
    }

    with open(args.output_report, "w") as f:
        json.dump(report_data, f, indent=2)

    # Print Clean Formatted Results Table
    print("\n" + "=" * 75)
    print(f"OFFICIAL TEST RESULTS: {args.env.upper()} RUN")
    print("=" * 75)
    print(f"  >>> VERIFIED ACCURACY     : {acc * 100:.2f}% <<<")
    print(f"  Precision                 : {p * 100:.2f}%")
    print(f"  Recall (Attack Catch Rate): {r * 100:.2f}%")
    print(f"  F1-Score                  : {f1 * 100:.2f}%")
    print(f"  False Positive Rate (FPR) : {fpr * 100:.2f}%")
    print(f"  ROC-AUC Score             : {roc_auc * 100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  - True Safe Prompts Correctly Allowed: {cm[0, 0]:,}")
    print(f"  - False Alarms (Safe marked Attack) : {cm[0, 1]:,}")
    print(f"  - Missed Attacks                    : {cm[1, 0]:,}")
    print(f"  - Attacks Successfully Blocked       : {cm[1, 1]:,}")
    print(f"\nResults saved to: {args.output_report}")

    # Manual Inspection: Print 10 Random Prompts
    print("\n" + "=" * 75)
    print("MANUAL PROMPT INSPECTION (Sample 10 Random Real Test Prompts):")
    print("=" * 75)
    test_df["predicted_label"] = preds
    test_df["attack_probability"] = probs

    sample = test_df.sample(n=10, random_state=42).reset_index(drop=True)
    for idx, row in sample.iterrows():
        is_correct = "CORRECT" if row["binary_label"] == row["predicted_label"] else "MISTAKE"
        true_tag = "ATTACK / MALICIOUS" if row["binary_label"] == 1 else "SAFE / BENIGN"
        pred_tag = "ATTACK / MALICIOUS" if row["predicted_label"] == 1 else "SAFE / BENIGN"
        print(f"\nSample #{idx + 1} | [{is_correct}] (Prob: {row['attack_probability']*100:.1f}%)")
        print(f"  Prompt Text : \"{row['head_tail_prompt'][:95]}...\"")
        print(f"  True Label  : {true_tag} ({row['binary_label']})")
        print(f"  Model Guess : {pred_tag} ({row['predicted_label']})")
        print("-" * 75)

    print("=" * 75)
    print("EVALUATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
