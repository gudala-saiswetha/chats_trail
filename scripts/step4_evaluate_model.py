#!/usr/bin/env python3
"""
Step 4: Comprehensive Multi-Task Evaluation & Robustness Stress Testing
Evaluates:
1. Binary Detection Accuracy, Precision, Recall, F1, ROC-AUC, FPR
2. Optimal Decision Threshold Calibration (Targeting FPR <= 1.5%)
3. Attack Category Classification Performance
4. Risk Severity Level Classification Performance
5. Adversarial Robustness under Homoglyph & Obfuscation Attacks (Objective 5)
6. Latency and Throughput Benchmarking
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from transformers import AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hls_guard import GatedHLSGuard
from models.adversarial_augmenter import AdversarialAugmenter
from models.dataset_loader import (
    build_pattern_matrix,
    PatternScaler,
    HLSGuardDataset
)

CATEGORY_NAMES = ["benign_control", "direct_injection", "jailbreak", "system_prompt_extraction"]
RISK_NAMES = ["none", "low", "medium", "high", "critical"]

def evaluate_test_set(args):
    print("=" * 65)
    print("STEP 4: COMPREHENSIVE MULTI-TASK & ADVERSARIAL EVALUATION")
    print("=" * 65)

    device = torch.device(
        args.device if args.device != "auto" else (
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
    )
    print(f"Evaluation Device: {device}")
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    # 1. Load Test Split
    print(f"\n[1/5] Loading test dataset: {args.test_path} ...")
    test_df = pd.read_parquet(args.test_path)
    if args.sample_size and len(test_df) > args.sample_size:
        print(f"Sampling {args.sample_size:,} records for fast evaluation...")
        test_df = test_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
    print(f"Evaluating {len(test_df):,} test records.")

    # 2. Load Pattern Scaler & Model Checkpoint
    scaler_path = os.path.join(os.path.dirname(args.checkpoint), "pattern_scaler.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = "models/pattern_scaler.pkl"
    print(f"Loading pattern scaler from: {scaler_path} ...")
    scaler = PatternScaler(scaler_path=scaler_path)

    test_raw_patterns = build_pattern_matrix(test_df, text_column="head_tail_prompt")
    test_patterns_scaled = scaler.load_and_transform(test_raw_patterns)

    # Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GatedHLSGuard(
        model_name=args.model_name,
        num_pattern_features=14,
        num_categories=4,
        num_risk_levels=5,
        fusion_dim=256,
        dropout=0.0
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from checkpoint: {args.checkpoint} ...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Notice: Checkpoint not found at {args.checkpoint}. Evaluating initialized model weights.")

    model.eval()

    # 3. Batch Inference on Test Set
    print("\n[2/5] Running multi-task test inference...")
    batch_size = args.batch_size
    binary_probs = []
    category_preds = []
    risk_preds = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(test_df), batch_size):
            end = min(start + batch_size, len(test_df))
            batch_texts = test_df.iloc[start:end]["head_tail_prompt"].tolist()

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            pat_tensor = torch.tensor(test_patterns_scaled[start:end], dtype=torch.float32, device=device)

            outputs = model(input_ids, attention_mask, pat_tensor)

            # Binary probabilities (softmax on malicious class)
            probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1].cpu().numpy()
            binary_probs.extend(probs)

            # Category and risk predictions
            c_pred = outputs["category_logits"].argmax(dim=-1).cpu().numpy()
            r_pred = outputs["risk_logits"].argmax(dim=-1).cpu().numpy()
            category_preds.extend(c_pred)
            risk_preds.extend(r_pred)

    total_time = time.perf_counter() - t0
    latency_ms = (total_time * 1000) / len(test_df)
    throughput = len(test_df) / max(total_time, 0.001)

    binary_probs = np.array(binary_probs)
    category_preds = np.array(category_preds)
    risk_preds = np.array(risk_preds)

    binary_trues = test_df["binary_label"].to_numpy()
    category_trues = test_df["category_label"].to_numpy()
    risk_trues = test_df["risk_label"].to_numpy()

    # 4. Binary Threshold Calibration (Objective 4: Suppress FPR <= 1.5%)
    print("\n[3/5] Calibrating decision thresholds for ultra-low False Positive Rate (FPR <= 1.5%)...")
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    threshold_results = []
    optimal_threshold = 0.50
    min_fpr_above_target_recall = float("inf")

    for th in thresholds:
        preds = (binary_probs >= th).astype(int)
        acc = accuracy_score(binary_trues, preds)
        p, r, f1, _ = precision_recall_fscore_support(binary_trues, preds, average="binary", zero_division=0)
        cm = confusion_matrix(binary_trues, preds, labels=[0, 1])
        tn, fp = cm[0, 0], cm[0, 1]
        fpr = fp / max(tn + fp, 1)

        threshold_results.append({
            "threshold": th,
            "accuracy": round(acc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4)
        })

        if fpr <= 0.015 and r >= 0.90:
            optimal_threshold = th

    th_df = pd.DataFrame(threshold_results)
    print("\nTHRESHOLD CALIBRATION TABLE:")
    print(th_df.to_string(index=False))

    # Standard metrics at 0.50 and optimal threshold
    standard_preds = (binary_probs >= 0.50).astype(int)
    roc_auc = roc_auc_score(binary_trues, binary_probs) if len(np.unique(binary_trues)) > 1 else 0.5

    # 5. Multi-Class Reports (Category & Risk)
    print("\n[4/5] Multi-Class Classification Reports...")
    cat_report = classification_report(
        category_trues,
        category_preds,
        labels=[0, 1, 2, 3],
        target_names=CATEGORY_NAMES,
        output_dict=True,
        zero_division=0
    )
    risk_report = classification_report(
        risk_trues,
        risk_preds,
        labels=[0, 1, 2, 3, 4],
        target_names=RISK_NAMES,
        output_dict=True,
        zero_division=0
    )

    print("\nAttack Category Breakdown:")
    for cat in CATEGORY_NAMES:
        print(f"  - {cat:25s}: Precision={cat_report[cat]['precision']:.4f}, Recall={cat_report[cat]['recall']:.4f}, F1={cat_report[cat]['f1-score']:.4f}")

    print("\nRisk Severity Breakdown:")
    for risk in RISK_NAMES:
        print(f"  - {risk:12s}: Precision={risk_report[risk]['precision']:.4f}, Recall={risk_report[risk]['recall']:.4f}, F1={risk_report[risk]['f1-score']:.4f}")

    # 6. Adversarial Robustness Stress Test (Objective 5: Unseen / Obfuscated Attacks)
    print("\n[5/5] Running Adversarial Robustness Stress Test on modified attacks...")
    augmenter = AdversarialAugmenter(p_homoglyph=0.8, p_leet=0.8, p_delimiter=0.5, p_casing=0.6)

    # Sample 500 malicious test prompts and perturb them
    mal_mask = test_df["is_malicious"] == True
    mal_df = test_df[mal_mask].head(500).copy()
    perturbed_texts = [augmenter.augment(t, is_malicious=True) for t in mal_df["prompt_text"]]

    # Evaluate perturbed prompts
    perturbed_patterns = build_pattern_matrix(pd.DataFrame({"head_tail_prompt": perturbed_texts}), text_column="head_tail_prompt")
    perturbed_patterns_scaled = scaler.load_and_transform(perturbed_patterns)

    pert_encoded = tokenizer(
        perturbed_texts,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        pert_out = model(
            pert_encoded["input_ids"].to(device),
            pert_encoded["attention_mask"].to(device),
            torch.tensor(perturbed_patterns_scaled, dtype=torch.float32, device=device)
        )
        pert_probs = torch.softmax(pert_out["binary_logits"], dim=-1)[:, 1].cpu().numpy()
        perturbed_malicious_recall = float((pert_probs >= 0.50).mean())

    print(f"\nAdversarial Robustness Stress Test:")
    print(f"  - Clean Malicious Prompt Recall    : {(binary_probs[binary_trues == 1] >= 0.50).mean():.4f}")
    print(f"  - Obfuscated / Perturbed Recall    : {perturbed_malicious_recall:.4f}")

    # 7. Compile Final Comprehensive Evaluation Report
    final_report = {
        "timestamp": "2026-10-23",
        "checkpoint": args.checkpoint,
        "test_samples_evaluated": len(test_df),
        "inference_benchmark": {
            "latency_ms_per_prompt": round(latency_ms, 2),
            "throughput_prompts_per_sec": round(throughput, 2),
            "device": str(device)
        },
        "binary_classification": {
            "accuracy_default_threshold": round(accuracy_score(binary_trues, standard_preds), 4),
            "roc_auc": round(roc_auc, 4),
            "optimal_calibrated_threshold": optimal_threshold,
            "threshold_sweep": threshold_results
        },
        "category_classification": {
            "accuracy": round(accuracy_score(category_trues, category_preds), 4),
            "macro_f1": round(precision_recall_fscore_support(category_trues, category_preds, average="macro", zero_division=0)[2], 4),
            "class_breakdown": {cat: cat_report[cat] for cat in CATEGORY_NAMES}
        },
        "risk_classification": {
            "accuracy": round(accuracy_score(risk_trues, risk_preds), 4),
            "macro_f1": round(precision_recall_fscore_support(risk_trues, risk_preds, average="macro", zero_division=0)[2], 4),
            "critical_risk_recall": round(risk_report["critical"]["recall"], 4),
            "class_breakdown": {risk: risk_report[risk] for risk in RISK_NAMES}
        },
        "adversarial_robustness_stress_test": {
            "clean_malicious_recall": round(float((binary_probs[binary_trues == 1] >= 0.50).mean()), 4),
            "obfuscated_perturbed_recall": round(perturbed_malicious_recall, 4),
            "retention_rate": round(perturbed_malicious_recall / max(float((binary_probs[binary_trues == 1] >= 0.50).mean()), 0.001), 4)
        }
    }

    with open(args.output_report, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nEvaluation Report successfully saved to: {args.output_report}")
    print("=" * 65)
    print("STEP 4 (PART A): COMPREHENSIVE MULTI-TASK EVALUATION COMPLETE")
    print("=" * 65)
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: HLS-Guard Comprehensive Evaluation")
    parser.add_argument("--test_path", type=str, default="dataset/cleaned/test.parquet")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_hls_guard.pt")
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_report", type=str, default="benchmarks/hls_guard_evaluation_report.json")

    args = parser.parse_args()
    evaluate_test_set(args)

