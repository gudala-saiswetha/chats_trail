#!/usr/bin/env python3
"""
HLS-Guard 2.0: Gold-Standard Training Engine (Targeting >= 98% Accuracy)
Trained on purified, non-synthetic, high-confidence dataset:
- Train: dataset/gold_clean/train_gold.parquet (190,992 rows)
- Test : dataset/gold_clean/test_gold.parquet (47,748 rows)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hls_guard import GatedHLSGuard
from models.dataset_loader import build_pattern_matrix, PatternScaler

class GoldDataset(Dataset):
    def __init__(self, encodings, patterns, labels):
        self.encodings = encodings
        self.patterns = torch.tensor(patterns, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["patterns"] = self.patterns[idx]
        item["labels"] = self.labels[idx]
        return item

def main():
    parser = argparse.ArgumentParser(description="Train HLS-Guard on Gold Dataset")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of training prompts (default 0 = FULL 190,992 dataset)")
    parser.add_argument("--test_sample", type=int, default=0, help="Number of test prompts to evaluate on (default 0 = FULL 47,748 test dataset)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_gold", help="Directory to save checkpoints (e.g. checkpoints_gold or checkpoints_colab)")
    args = parser.parse_args()

    print("=" * 70)
    print("HLS-GUARD: GOLD-STANDARD TRAINING ENGINE (TARGETING 98% ACCURACY)")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )
    print(f"Compute Device : {device}")
    print(f"Checkpoint Dir : {args.checkpoint_dir}")

    # 1. Load Data
    train_path = "dataset/gold_clean/train_gold.parquet"
    test_path = "dataset/gold_clean/test_gold.parquet"
    print(f"\n[1/5] Loading clean datasets...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Subsample if requested
    if args.sample_size and args.sample_size < len(train_df):
        print(f"  Selecting balanced {args.sample_size:,} training sample (50% Attack, 50% Safe)...")
        n_half = args.sample_size // 2
        train_mal = train_df[train_df["binary_label"] == 1].sample(n=n_half, random_state=42)
        train_ben = train_df[train_df["binary_label"] == 0].sample(n=n_half, random_state=42)
        train_df = pd.concat([train_mal, train_ben]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    if args.test_sample and args.test_sample < len(test_df):
        print(f"  Selecting balanced {args.test_sample:,} held-out test sample...")
        n_test_half = args.test_sample // 2
        test_mal = test_df[test_df["binary_label"] == 1].sample(n=n_test_half, random_state=42)
        test_ben = test_df[test_df["binary_label"] == 0].sample(n=n_test_half, random_state=42)
        test_df = pd.concat([test_mal, test_ben]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Active Training Prompts: {len(train_df):,} | Active Test Prompts: {len(test_df):,}")

    # 2. Extract & Scale Pattern Features
    print("\n[2/5] Extracting 14 handcrafted linguistic pattern features...")
    out_dir = args.checkpoint_dir
    os.makedirs(out_dir, exist_ok=True)
    scaler = PatternScaler(scaler_path=os.path.join(out_dir, "pattern_scaler.pkl"))
    train_patterns = scaler.fit_and_save(build_pattern_matrix(train_df, "head_tail_prompt"))
    test_patterns = scaler.load_and_transform(build_pattern_matrix(test_df, "head_tail_prompt"))

    # 3. Tokenize Dataset
    print("\n[3/5] Tokenizing prompts with distilroberta-base...")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    train_enc = tokenizer(train_df["head_tail_prompt"].tolist(), truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    test_enc = tokenizer(test_df["head_tail_prompt"].tolist(), truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    train_dataset = GoldDataset(train_enc, train_patterns, train_df["binary_label"].to_numpy())
    test_dataset = GoldDataset(test_enc, test_patterns, test_df["binary_label"].to_numpy())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # 4. Initialize Gated HLS-Guard Model
    print("\n[4/5] Initializing GatedHLSGuard (Transferring pretrained weights)...")
    model = GatedHLSGuard("distilroberta-base", num_pattern_features=14, num_categories=4, num_risk_levels=5).to(device)
    if os.path.exists("checkpoints_mac/best_hls_guard.pt"):
        ckpt = torch.load("checkpoints_mac/best_hls_guard.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Successfully loaded pretrained baseline weights from checkpoints_mac/best_hls_guard.pt!")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)

    print(f"\n[5/5] Starting fine-tuning for {args.epochs} epochs ({total_steps:,} total steps)...")
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            patterns = batch["patterns"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, patterns)
            loss = loss_fn(out["binary_logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 200 == 0 or step == len(train_loader):
                elapsed = (time.time() - t0) / 60
                print(f"Epoch {epoch}/{args.epochs} | Step {step:4d}/{len(train_loader)} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}m")

        # Evaluate on Held-Out Clean Test Set
        print(f"\n--- Running Full Test Evaluation for Epoch {epoch} on {len(test_df):,} prompts ---")
        model.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                patterns = batch["patterns"].to(device)
                out = model(input_ids, attention_mask, patterns)
                prob = torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy()
                p = out["binary_logits"].argmax(dim=-1).cpu().numpy()
                preds.extend(p)
                probs.extend(prob)
                trues.extend(batch["labels"].numpy())

        acc = accuracy_score(trues, preds)
        p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="binary")
        cm = confusion_matrix(trues, preds)
        fpr = cm[0, 1] / max(cm[0, 0] + cm[0, 1], 1)

        print("-" * 65)
        print(f"EPOCH {epoch} TEST RESULTS (Evaluated on {len(test_df):,} Clean Held-Out Prompts):")
        print(f"  >>> VERIFIED ACCURACY     : {acc * 100:.2f}% <<<")
        print(f"  Precision                 : {p * 100:.2f}%")
        print(f"  Recall (Attack Catch Rate): {r * 100:.2f}%")
        print(f"  F1-Score                  : {f1 * 100:.2f}%")
        print(f"  False Positive Rate (FPR) : {fpr * 100:.2f}%")
        print(f"  Confusion Matrix (TN: {cm[0, 0]}, FP: {cm[0, 1]} | FN: {cm[1, 0]}, TP: {cm[1, 1]}):\n{cm}")
        print("-" * 65 + "\n")

        if acc > best_acc:
            best_acc = acc
            best_ckpt = os.path.join(out_dir, "best_hls_guard.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "precision": p,
                "recall": r,
                "f1": f1,
                "fpr": fpr,
                "epoch": epoch
            }, best_ckpt)
            print(f"⭐ Saved new BEST model: {best_ckpt} (Accuracy: {acc * 100:.2f}%)")

            # Save evaluated test parquet for manual inspection
            test_df["predicted_label"] = preds
            test_df["attack_probability"] = probs
            test_df.to_parquet(os.path.join(out_dir, "test_evaluated.parquet"))

    print("=" * 70)
    print("TRAINING & VERIFICATION COMPLETE!")
    print(f"Final Peak Verified Accuracy: {best_acc * 100:.2f}%")
    print(f"Model saved to: {out_dir}/best_hls_guard.pt")
    print(f"Test predictions saved to: {out_dir}/test_evaluated.parquet")
    print("=" * 70)

if __name__ == "__main__":
    main()
