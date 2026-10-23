#!/usr/bin/env python3
"""
HLS-Guard 2.0 Unified Multi-Task Adversarial Training Engine
Implements:
1. Online Adversarial Augmentation (Homoglyphs, Leetspeak, Tags)
2. Asymmetric Multi-Task Focal Loss for FPR Suppression (<1.5%) and Class Balance
3. Cosine Annealing Learning Rate Scheduler with Warmup
4. Automated Checkpointing & Validation
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.hls_guard import GatedHLSGuard
from models.adversarial_augmenter import AdversarialAugmenter
from models.losses import MultiTaskHLSGuardLoss
from models.dataset_loader import (
    build_pattern_matrix,
    PatternScaler,
    HLSGuardDataset
)

def evaluate_validation(model, val_loader, loss_fn, device, max_batches: int = None):
    """Evaluates the model on validation data across all 3 heads and computes comprehensive metrics."""
    model.eval()
    total_val_loss = 0.0
    all_bin_preds, all_bin_trues = [], []
    all_cat_preds, all_cat_trues = [], []
    all_risk_preds, all_risk_trues = [], []

    batch_count = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pattern_features = batch["pattern_features"].to(device)
            bin_targets = batch["binary_labels"].to(device)
            cat_targets = batch["category_labels"].to(device)
            risk_targets = batch["risk_labels"].to(device)

            outputs = model(input_ids, attention_mask, pattern_features)
            loss_dict = loss_fn(
                outputs["binary_logits"],
                outputs["category_logits"],
                outputs["risk_logits"],
                bin_targets,
                cat_targets,
                risk_targets
            )
            total_val_loss += loss_dict["total_loss"].item()

            # Predictions
            bin_pred = outputs["binary_logits"].argmax(dim=-1).cpu().numpy()
            cat_pred = outputs["category_logits"].argmax(dim=-1).cpu().numpy()
            risk_pred = outputs["risk_logits"].argmax(dim=-1).cpu().numpy()

            all_bin_preds.extend(bin_pred)
            all_bin_trues.extend(bin_targets.cpu().numpy())
            all_cat_preds.extend(cat_pred)
            all_cat_trues.extend(cat_targets.cpu().numpy())
            all_risk_preds.extend(risk_pred)
            all_risk_trues.extend(risk_targets.cpu().numpy())

            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break

    # Metrics computation
    bin_acc = accuracy_score(all_bin_trues, all_bin_preds)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(all_bin_trues, all_bin_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_bin_trues, all_bin_preds, labels=[0, 1])
    # FPR = FP / (FP + TN)
    tn, fp = cm[0, 0], cm[0, 1]
    fpr = fp / max(tn + fp, 1)

    cat_acc = accuracy_score(all_cat_trues, all_cat_preds)
    cat_macro_f1 = precision_recall_fscore_support(all_cat_trues, all_cat_preds, average="macro", zero_division=0)[2]

    risk_acc = accuracy_score(all_risk_trues, all_risk_preds)
    risk_macro_f1 = precision_recall_fscore_support(all_risk_trues, all_risk_preds, average="macro", zero_division=0)[2]

    model.train()
    return {
        "val_loss": total_val_loss / max(batch_count, 1),
        "binary_accuracy": bin_acc,
        "binary_precision": bin_p,
        "binary_recall": bin_r,
        "binary_f1": bin_f1,
        "binary_fpr": fpr,
        "category_accuracy": cat_acc,
        "category_macro_f1": cat_macro_f1,
        "risk_accuracy": risk_acc,
        "risk_macro_f1": risk_macro_f1,
        "confusion_matrix": cm.tolist()
    }


def train_hls_guard(args):
    print("=" * 65)
    print("HLS-GUARD 2.0: MULTI-TASK ADVERSARIAL TRAINING ENGINE")
    print("=" * 65)

    device = torch.device(
        args.device if args.device != "auto" else (
            "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
    )
    print(f"Compute Device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load Data
    print(f"\nLoading training data: {args.data_path} ...")
    train_df = pd.read_parquet(args.data_path)
    val_df = pd.read_parquet(args.val_path)
    print(f"Train records: {len(train_df):,} | Val records: {len(val_df):,}")

    # 2. Tokenizer & Scaler
    print(f"\nInitializing tokenizer: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    scaler = PatternScaler(scaler_path=os.path.join(args.checkpoint_dir, "pattern_scaler.pkl"))

    # Fit scaler on train pattern features
    print("Fitting pattern feature scaler...")
    train_raw_patterns = build_pattern_matrix(train_df, text_column="head_tail_prompt")
    train_patterns_scaled = scaler.fit_and_save(train_raw_patterns)

    val_raw_patterns = build_pattern_matrix(val_df, text_column="head_tail_prompt")
    val_patterns_scaled = scaler.load_and_transform(val_raw_patterns)

    # Pre-tokenize validation set for speed
    print("Tokenizing validation split...")
    val_encoded = tokenizer(
        val_df["head_tail_prompt"].tolist(),
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt"
    )
    val_dataset = HLSGuardDataset(
        input_ids=val_encoded["input_ids"],
        attention_mask=val_encoded["attention_mask"],
        pattern_features=val_patterns_scaled,
        binary_labels=val_df["binary_label"].to_numpy(),
        category_labels=val_df["category_label"].to_numpy(),
        risk_labels=val_df["risk_label"].to_numpy(),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # 3. Model & Loss & Optimizer
    print(f"\nInitializing GatedHLSGuard model...")
    model = GatedHLSGuard(
        model_name=args.model_name,
        num_pattern_features=14,
        num_categories=4,
        num_risk_levels=5,
        fusion_dim=256,
        dropout=args.dropout
    ).to(device)

    loss_fn = MultiTaskHLSGuardLoss(
        alpha_benign=args.alpha_benign,
        gamma=2.0,
        w_binary=1.0,
        w_category=0.5,
        w_risk=0.5
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    augmenter = AdversarialAugmenter(p_homoglyph=0.30, p_leet=0.30, p_delimiter=0.20, p_casing=0.35)

    # Calculate steps
    steps_per_epoch = int(np.ceil(len(train_df) / args.batch_size))
    total_steps = args.max_steps if args.max_steps else (steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"Total planned steps: {total_steps:,} (Warmup: {warmup_steps:,} steps)")
    print(f"Batch size: {args.batch_size} | Learning rate: {args.lr}")

    # Resume if requested
    best_val_score = -float("inf")
    global_step = 0
    if args.resume and os.path.exists(os.path.join(args.checkpoint_dir, "latest_hls_guard.pt")):
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "latest_hls_guard.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        best_val_score = ckpt.get("best_val_score", -float("inf"))
        print(f"Resumed from global step {global_step} (Best Val Score: {best_val_score:.4f})")

    # 4. Training Loop
    print("\nStarting training loop...")
    model.train()
    start_time = time.time()
    running_loss = 0.0

    # Generator for batching with on-the-fly augmentation
    num_samples = len(train_df)
    indices = np.arange(num_samples)

    while global_step < total_steps:
        np.random.shuffle(indices)
        for i in range(0, num_samples, args.batch_size):
            if global_step >= total_steps:
                break

            batch_idx = indices[i:i + args.batch_size]
            batch_texts = train_df.iloc[batch_idx]["head_tail_prompt"].tolist()
            batch_is_mal = train_df.iloc[batch_idx]["is_malicious"].tolist()
            batch_patterns = train_patterns_scaled[batch_idx]
            batch_bin = train_df.iloc[batch_idx]["binary_label"].to_numpy()
            batch_cat = train_df.iloc[batch_idx]["category_label"].to_numpy()
            batch_risk = train_df.iloc[batch_idx]["risk_label"].to_numpy()

            # Dynamic online adversarial augmentation (25% probability on malicious)
            if np.random.rand() < args.adversarial_prob:
                batch_texts = [
                    augmenter.augment(t, is_mal) if is_mal and np.random.rand() < 0.6 else t
                    for t, is_mal in zip(batch_texts, batch_is_mal)
                ]

            # Tokenize batch
            encoded_batch = tokenizer(
                batch_texts,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoded_batch["input_ids"].to(device)
            attention_mask = encoded_batch["attention_mask"].to(device)
            pattern_t = torch.tensor(batch_patterns, dtype=torch.float32, device=device)
            bin_t = torch.tensor(batch_bin, dtype=torch.long, device=device)
            cat_t = torch.tensor(batch_cat, dtype=torch.long, device=device)
            risk_t = torch.tensor(batch_risk, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, pattern_t)

            losses = loss_fn(
                outputs["binary_logits"],
                outputs["category_logits"],
                outputs["risk_logits"],
                bin_t,
                cat_t,
                risk_t
            )
            total_loss = losses["total_loss"]
            total_loss.backward()

            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += total_loss.item()

            # Log periodic progress
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(
                    f"Step {global_step:5d}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} (Bin: {losses['binary_loss'].item():.3f}, "
                    f"Cat: {losses['category_loss'].item():.3f}, Risk: {losses['risk_loss'].item():.3f}) | "
                    f"LR: {current_lr:.2e} | Elapsed: {elapsed/60:.1f}m"
                )

            # Evaluation Interval
            if global_step % args.eval_every == 0 or global_step == total_steps:
                print(f"\n--- Validation Diagnostic at Step {global_step} ---")
                val_results = evaluate_validation(model, val_loader, loss_fn, device, max_batches=args.val_batches)
                print(
                    f"  Binary Acc : {val_results['binary_accuracy']:.4f} | "
                    f"Binary F1  : {val_results['binary_f1']:.4f} | "
                    f"Binary FPR : {val_results['binary_fpr']:.4f} (Target <= 0.015)"
                )
                print(
                    f"  Category F1: {val_results['category_macro_f1']:.4f} | "
                    f"Risk F1    : {val_results['risk_macro_f1']:.4f}"
                )

                # Score combines high binary F1 with penalty for False Positives
                val_score = val_results["binary_f1"] - (2.0 * val_results["binary_fpr"])

                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_path = os.path.join(args.checkpoint_dir, "best_hls_guard.pt")
                    torch.save({
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_results,
                        "best_val_score": best_val_score,
                        "model_name": args.model_name
                    }, best_path)
                    print(f"  ⭐ Saved new BEST model: {best_path} (Score: {best_val_score:.4f})")

                # Save latest checkpoint
                latest_path = os.path.join(args.checkpoint_dir, "latest_hls_guard.pt")
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_score": best_val_score,
                    "val_metrics": val_results,
                    "model_name": args.model_name
                }, latest_path)

    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print(f"Total Steps: {global_step} | Total Time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"Best Validation Score: {best_val_score:.4f}")
    print(f"Best Checkpoint: {os.path.join(args.checkpoint_dir, 'best_hls_guard.pt')}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HLS-Guard 2.0 Adversarial Training")
    parser.add_argument("--data_path", type=str, default="dataset/cleaned/train_subset_80k.parquet")
    parser.add_argument("--val_path", type=str, default="dataset/cleaned/val.parquet")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha_benign", type=float, default=0.70)
    parser.add_argument("--adversarial_prob", type=float, default=0.25)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    train_hls_guard(args)

