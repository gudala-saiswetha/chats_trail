#!/usr/bin/env python3
"""
Interactive & Transparent Verification Script for HLS-Guard.
Allows manual and automated inspection of test prompts and ground truth labels.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def verify():
    print("=" * 75)
    print("      HLS-GUARD: INDEPENDENT & TRANSPARENT ACCURACY VERIFIER")
    print("=" * 75)

    test_file = "checkpoints_gold/test_evaluated.parquet"
    if not os.path.exists(test_file):
        test_file = "checkpoints_98/test_evaluated_4k.parquet"

    if not os.path.exists(test_file):
        print(f"Notice: {test_file} not found yet. Please run the training script first:")
        print("  python3 scripts/train_gold_model.py")
        return

    df = pd.read_parquet(test_file)
    y_true = df["binary_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
    fpr = cm[0, 1] / max(cm[0, 0] + cm[0, 1], 1)

    print(f"\nTotal Real Test Prompts Evaluated: {len(df):,}")
    print(f"  >>> VERIFIED ACCURACY    : {acc * 100:.2f}% <<<")
    print(f"  >>> PRECISION            : {p * 100:.2f}%")
    print(f"  >>> RECALL (CATCH RATE)  : {r * 100:.2f}%")
    print(f"  >>> F1-SCORE             : {f1 * 100:.2f}%")
    print(f"  >>> FALSE POSITIVE RATE  : {fpr * 100:.2f}%")
    print(f"\nConfusion Matrix (True Safe: {cm[0, 0]}, False Alarms: {cm[0, 1]} | Missed Attacks: {cm[1, 0]}, Caught Attacks: {cm[1, 1]}):")
    print(cm)

    print("\n" + "=" * 75)
    print("MANUAL PROMPT INSPECTION (Sample 10 Random Real Test Prompts):")
    print("=" * 75)

    sample = df.sample(n=10, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
    for i, row in sample.iterrows():
        is_correct = "✅ CORRECT" if row["binary_label"] == row["predicted_label"] else "❌ MISTAKE"
        true_text = "ATTACK / MALICIOUS" if row["binary_label"] == 1 else "SAFE / BENIGN"
        pred_text = "ATTACK / MALICIOUS" if row["predicted_label"] == 1 else "SAFE / BENIGN"
        prob_str = f" (Prob: {row['attack_probability']*100:.1f}%)" if 'attack_probability' in row else ""
        print(f"\nSample #{i + 1} | Result: {is_correct}{prob_str}")
        print(f"  Prompt Text : \"{row['head_tail_prompt'][:100]}...\"")
        print(f"  True Label  : {true_text} ({row['binary_label']})")
        print(f"  Model Guess : {pred_text} ({row['predicted_label']})")
        print("-" * 75)

if __name__ == "__main__":
    verify()
