#!/usr/bin/env python3
"""
Step 1: Dataset Sanitization, Conflict Reconciliation, and Compound Stratification Pipeline
Part of HLS-Guard 2.0 (Vectorized & Highly Optimized)
"""

import json
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "dataset/guardrailer_dataset_v1.parquet"
OUTPUT_DIR = "dataset/cleaned"
REPORT_PATH = os.path.join(OUTPUT_DIR, "cleaning_report.json")

CATEGORY_TO_ID = {
    "benign_control": 0,
    "direct_injection": 1,
    "jailbreak": 2,
    "system_prompt_extraction": 3,
}

RISK_TO_ID = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

INJECTION_KEYWORDS = [
    r"\bignore (all )?(previous|prior|above) instructions\b",
    r"\bdisregard (all )?(previous|prior|above) instructions\b",
    r"\boverride (the )?(system|previous) instructions\b",
    r"\breveal (the )?(system prompt|hidden prompt)\b",
    r"\bshow (me )?(the )?(system prompt|hidden instructions)\b",
    r"\brepeat (the )?(system prompt|instructions)\b",
    r"\byou are now\b",
    r"\bact as (an? )?(unrestricted|different|new)\b",
    r"\bdeveloper mode\b",
    r"\bdan mode\b",
    r"\bbase64\b",
    r"\bdecode this\b",
]

def contains_injection_pattern(text: str) -> bool:
    lower = str(text).lower()
    return any(re.search(pat, lower, re.I) for pat in INJECTION_KEYWORDS)

def is_corrupted_or_garbage(text: str) -> bool:
    text_str = str(text).strip()
    if len(text_str) < 5:
        return True
    if "\ufffd" in text_str or "\x00" in text_str:
        return True
    unusual_ratio = sum(c in "^~`|\\/*" for c in text_str) / max(len(text_str), 1)
    if unusual_ratio > 0.35 and len(text_str) < 30:
        return True
    if len(text_str) < 15 and not any(c.isalnum() for c in text_str):
        return True
    return False

def head_tail_format(text: str, max_words_each: int = 120) -> str:
    words = str(text).split()
    if len(words) <= (max_words_each * 2):
        return str(text)
    head = " ".join(words[:max_words_each])
    tail = " ".join(words[-max_words_each:])
    return f"{head} ... [MID_TEXT_OMITTED] ... {tail}"

def run_step1_cleaning():
    print("=" * 65)
    print("STEP 1: DATASET SANITIZATION & RECONCILIATION PIPELINE")
    print("=" * 65)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_DATA_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading raw dataset: {RAW_DATA_PATH} ...")
    df = pd.read_parquet(RAW_DATA_PATH)
    initial_count = len(df)
    print(f"Loaded {initial_count:,} records.")

    # 1. Filter corrupted / byte garbage
    print("\n[1/6] Filtering corrupted and garbage byte rows...")
    is_garbage = df["prompt_text"].apply(is_corrupted_or_garbage)
    garbage_count = int(is_garbage.sum())
    df = df[~is_garbage].copy()
    print(f"Removed {garbage_count:,} corrupted/garbage records.")

    # 2. Vectorized reconciliation of duplicates and conflicts
    print("\n[2/6] Detecting and resolving conflicting labels across merged datasets...")
    df["norm_text"] = (
        df["prompt_text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    is_dupe = df.duplicated(subset=["norm_text"], keep=False)
    unique_df = df[~is_dupe].copy()
    dupe_df = df[is_dupe].copy()
    print(f"Unique texts: {len(unique_df):,}, Duplicated rows across sources: {len(dupe_df):,}")

    resolved_dupes = []
    conflict_count = 0

    for norm_t, group in dupe_df.groupby("norm_text"):
        if group["is_malicious"].nunique() > 1:
            conflict_count += 1
            # If explicit injection signature exists, pick the highest risk malicious row
            if contains_injection_pattern(norm_t):
                mal_rows = group[group["is_malicious"] == True]
                if len(mal_rows) > 0:
                    chosen = mal_rows.sort_values(
                        by="risk_level",
                        key=lambda col: col.map(RISK_TO_ID).fillna(0),
                        ascending=False
                    ).iloc[0]
                    resolved_dupes.append(chosen)
                    continue
            else:
                ben_rows = group[group["is_malicious"] == False]
                if len(ben_rows) > 0:
                    resolved_dupes.append(ben_rows.iloc[0])
                    continue
        # Non-conflicting duplicate: keep first
        resolved_dupes.append(group.iloc[0])

    resolved_dupe_df = pd.DataFrame(resolved_dupes)
    clean_df = pd.concat([unique_df, resolved_dupe_df], ignore_index=True)
    print(f"Resolved {conflict_count:,} conflicting groups.")
    print(f"Total deduplicated & reconciled records: {len(clean_df):,}")

    # 3. Label reconciliation verification
    print("\n[3/6] Enforcing strict label consistency rules...")
    benign_mask = clean_df["attack_category"] == "benign_control"
    clean_df.loc[benign_mask, "is_malicious"] = False
    clean_df.loc[benign_mask, "risk_level"] = "none"

    mal_none_mask = (clean_df["is_malicious"] == True) & (clean_df["risk_level"] == "none")
    clean_df.loc[mal_none_mask, "risk_level"] = "low"

    # 4. Map integer targets
    print("\n[4/6] Mapping integer targets (Binary, Category, Risk)...")
    clean_df["binary_label"] = clean_df["is_malicious"].astype(int)
    clean_df["category_label"] = clean_df["attack_category"].map(CATEGORY_TO_ID).astype(int)
    clean_df["risk_label"] = clean_df["risk_level"].map(RISK_TO_ID).astype(int)

    # 5. Compound Stratification Key & Head-Tail Preserver
    print("\n[5/6] Generating compound stratification key & head-tail prompts...")
    clean_df["compound_strat"] = (
        clean_df["attack_category"].astype(str) + "_" + clean_df["risk_level"].astype(str)
    )
    clean_df["head_tail_prompt"] = clean_df["prompt_text"].apply(head_tail_format)

    # 6. Stratified 80 / 10 / 10 Split
    print("\n[6/6] Executing stratified 80/10/10 split on compound key...")
    cols_to_save = [
        "id", "prompt_text", "head_tail_prompt", "is_malicious",
        "attack_category", "attack_technique", "risk_level",
        "binary_label", "category_label", "risk_label",
        "compound_strat", "source_dataset", "length_norm"
    ]
    final_df = clean_df[cols_to_save].copy()

    train_df, temp_df = train_test_split(
        final_df,
        test_size=0.20,
        random_state=42,
        stratify=final_df["compound_strat"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["compound_strat"]
    )

    # Verification assertions
    assert set(train_df["id"]).isdisjoint(val_df["id"]), "Train and Val IDs overlap!"
    assert set(train_df["id"]).isdisjoint(test_df["id"]), "Train and Test IDs overlap!"
    assert set(val_df["id"]).isdisjoint(test_df["id"]), "Val and Test IDs overlap!"

    # Save full splits
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path = os.path.join(OUTPUT_DIR, "test.parquet")

    print(f"Saving train split: {train_path} ({len(train_df):,} rows)...")
    train_df.to_parquet(train_path, index=False)

    print(f"Saving val split: {val_path} ({len(val_df):,} rows)...")
    val_df.to_parquet(val_path, index=False)

    print(f"Saving test split: {test_path} ({len(test_df):,} rows)...")
    test_df.to_parquet(test_path, index=False)

    # Create High-Density 80k Balanced Training Subset
    print("\nCreating high-density 80,000-sample balanced training subset (preserving 100% rare classes)...")
    rare_mask = train_df["risk_level"].isin(["critical", "medium"])
    rare_train = train_df[rare_mask]
    common_train = train_df[~rare_mask]

    target_common_samples = 80000 - len(rare_train)
    common_sampled, _ = train_test_split(
        common_train,
        train_size=target_common_samples,
        random_state=42,
        stratify=common_train["compound_strat"]
    )

    subset_80k = pd.concat([rare_train, common_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    subset_path = os.path.join(OUTPUT_DIR, "train_subset_80k.parquet")
    subset_80k.to_parquet(subset_path, index=False)
    print(f"Saved high-density training subset: {subset_path} ({len(subset_80k):,} rows).")

    # Generate cleaning report
    report = {
        "timestamp": "2026-10-23",
        "initial_raw_records": initial_count,
        "garbage_records_removed": garbage_count,
        "conflict_groups_resolved": conflict_count,
        "clean_records_total": len(final_df),
        "splits": {
            "train_full": len(train_df),
            "train_subset_80k": len(subset_80k),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "rare_class_counts_in_train_subset_80k": {
            "critical": int((subset_80k["risk_level"] == "critical").sum()),
            "medium": int((subset_80k["risk_level"] == "medium").sum()),
        },
        "compound_strat_groups": int(clean_df["compound_strat"].nunique())
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCleaning Report saved to {REPORT_PATH}:")
    print(json.dumps(report, indent=2))
    print("=" * 65)
    print("STEP 1 COMPLETE: Clean datasets ready for Phase 2.")
    print("=" * 65)

if __name__ == "__main__":
    run_step1_cleaning()

