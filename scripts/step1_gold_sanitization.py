#!/usr/bin/env python3
"""
Step 1: Gold-Standard Dataset Sanitization Pipeline
Removes:
1. Missing, null, and empty rows.
2. Non-ASCII foreign scripts (Thai, Cyrillic, Hindi, Kannada, Korean, Chinese, Arabic).
3. Non-English Latin texts (Tagalog, Cebuano, German, Bosnian).
4. Fake / Inverted Labels (e.g. recipe assistants labeled as attacks, animal welfare labeled as jailbreaks).
5. Duplicate contradictions (prompts with conflicting labels).
Produces:
- dataset/gold_clean/train_gold.parquet (80% balanced)
- dataset/gold_clean/test_gold.parquet (20% balanced)
- dataset/gold_clean/cleaning_report.json
"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np

def run_gold_sanitization():
    print("=" * 70)
    print("STEP 1: GOLD-STANDARD DATASET PURIFICATION & SANITIZATION")
    print("=" * 70)

    raw_path = "dataset/guardrailer_dataset_v1.parquet"
    out_dir = "dataset/gold_clean"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[1/6] Loading raw dataset: {raw_path} ...")
    df = pd.read_parquet(raw_path)
    total_raw = len(df)
    print(f"  Total raw records: {total_raw:,}")

    # Track removals
    audit_stats = {"total_raw_records": total_raw}

    # 1. Null / Empty filter
    valid_text_mask = df['prompt_text'].apply(lambda s: isinstance(s, str) and len(s.strip()) >= 15)
    dropped_null_short = total_raw - valid_text_mask.sum()
    df = df[valid_text_mask].copy()
    audit_stats["dropped_null_or_too_short"] = int(dropped_null_short)
    print(f"  [Removed] Null, empty, or < 15 character rows: {dropped_null_short:,}")

    # 2. Non-ASCII script filter (Thai, Hindi, Japanese, Cyrillic, Arabic, etc.)
    ascii_mask = df['prompt_text'].apply(lambda s: s.isascii())
    dropped_foreign_script = len(df) - ascii_mask.sum()
    df = df[ascii_mask].copy()
    audit_stats["dropped_non_ascii_scripts"] = int(dropped_foreign_script)
    print(f"  [Removed] Non-ASCII foreign scripts (Thai, Hindi, Cyrillic, CJK, etc.): {dropped_foreign_script:,}")

    # 3. Real English word filter (removes Latin foreign texts: German, Cebuano, Tagalog, Bosnian)
    common_eng = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'])
    def is_english(s):
        words = set(re.findall(r'[a-zA-Z]+', s.lower()))
        if len(words) < 4:
            return True
        return len(words.intersection(common_eng)) >= 2

    eng_mask = df['prompt_text'].apply(is_english)
    dropped_non_eng = len(df) - eng_mask.sum()
    df = df[eng_mask].copy()
    audit_stats["dropped_latin_foreign_languages"] = int(dropped_non_eng)
    print(f"  [Removed] Latin foreign languages (Cebuano, Tagalog, German, etc.): {dropped_non_eng:,}")

    # 4. Purge Fake / Inverted / Synthetic Keyword Dump Labels
    print("\n[2/6] Purging corrupted and inverted external dataset labels...")
    # A) In gabrielchua/system-prompt-leakage, all 71,000 system prompts were falsely marked as "malicious attacks"
    # Drop this entire corrupted subset
    dropped_gabriel = len(df[df['source_dataset'] == 'HF:gabrielchua/system-prompt-leakage'])
    df = df[df['source_dataset'] != 'HF:gabrielchua/system-prompt-leakage'].copy()
    audit_stats["dropped_fake_system_prompt_leakage"] = int(dropped_gabriel)
    print(f"  [Removed] gabrielchua fake attack labels (harmless system prompts): {dropped_gabriel:,}")

    # B) In llm_jailbreak_injection, rows where attack_category='jailbreak' and attack_technique='none' (225k rows)
    # had random statements about Buddhism, hippopotamuses, animal welfare blindly marked as jailbreaks
    dropped_keyword_dump = len(df[(df['source_dataset'] == 'llm_jailbreak_injection') & (df['attack_category'] == 'jailbreak') & (df['attack_technique'] == 'none')])
    df = df[~((df['source_dataset'] == 'llm_jailbreak_injection') & (df['attack_category'] == 'jailbreak') & (df['attack_technique'] == 'none'))].copy()
    audit_stats["dropped_unverified_keyword_dump_jailbreaks"] = int(dropped_keyword_dump)
    print(f"  [Removed] Unverified keyword-dump jailbreaks (animal welfare, Buddhism, etc.): {dropped_keyword_dump:,}")

    # 5. Extract Confirmed, High-Confidence Real Attacks and Safe Benign Prompts
    print("\n[3/6] Curating confirmed genuine attacks and genuine benign queries...")
    # Real Attacks
    real_mal = df[
        (df['source_dataset'].isin(['HF:Lakera/mosscap_prompt_injection', 'HF:aurora-m/adversarial-prompts'])) |
        ((df['source_dataset'] == 'llm_jailbreak_injection') & (df['attack_technique'].isin(['virtualization_roleplay', 'base64_encoding', 'few_shot_override', 'payload_splitting', 'hypothetical_scenario']))) |
        ((df['source_dataset'] == 'HF:ahsanayub/malicious-prompts') & (df['is_malicious'] == True) & (df['prompt_text'].str.lower().str.contains('ignore|disregard|secret key|bypass|pwned|override|reveal|system prompt', regex=True)))
    ].copy()
    real_mal['is_malicious'] = True
    real_mal['binary_label'] = 1

    # Real Benign Queries
    real_ben = df[
        (df['source_dataset'] == 'HF:quickium/prompt-security-v0') |
        ((df['source_dataset'] == 'llm_jailbreak_injection') & (df['attack_category'] == 'benign_control') & (df['attack_technique'] == 'none') & (df['is_malicious'] == False)) |
        ((df['source_dataset'] == 'HF:ahsanayub/malicious-prompts') & (df['is_malicious'] == False) & (~df['prompt_text'].str.lower().str.contains('ignore|disregard|secret key|bypass|pwned|override|reveal|system prompt', regex=True)))
    ].copy()
    real_ben['is_malicious'] = False
    real_ben['binary_label'] = 0

    print(f"  Candidate Real Attacks : {len(real_mal):,}")
    print(f"  Candidate Real Benign  : {len(real_ben):,}")

    curated = pd.concat([real_mal, real_ben]).reset_index(drop=True)

    # 6. Remove Contradictory Label Duplicates
    print("\n[4/6] Eliminating contradictory label collisions...")
    text_labels = curated.groupby('prompt_text')['binary_label'].nunique()
    contradictory_texts = set(text_labels[text_labels > 1].index)
    dropped_contradictions = len(curated[curated['prompt_text'].isin(contradictory_texts)])
    curated = curated[~curated['prompt_text'].isin(contradictory_texts)].copy()
    audit_stats["dropped_contradictory_label_rows"] = int(dropped_contradictions)
    print(f"  [Removed] Exact duplicate texts with conflicting labels: {dropped_contradictions:,}")

    # Remove exact duplicate texts to prevent data leakage
    pre_dedup = len(curated)
    curated = curated.drop_duplicates(subset=['prompt_text']).reset_index(drop=True)
    audit_stats["dropped_duplicate_prompts"] = int(pre_dedup - len(curated))
    print(f"  [Removed] Exact duplicate prompts: {pre_dedup - len(curated):,}")

    # Format head_tail_prompt preservation
    def make_head_tail(text, max_words=120):
        words = text.split()
        if len(words) <= max_words * 2:
            return text
        return " ".join(words[:max_words]) + " ... " + " ".join(words[-max_words:])

    curated['head_tail_prompt'] = curated['prompt_text'].apply(make_head_tail)

    # Align multi-task categories and risk levels
    category_map = {"benign_control": 0, "direct_injection": 1, "jailbreak": 2, "system_prompt_extraction": 3}
    curated['category_label'] = curated['attack_category'].map(lambda c: category_map.get(c, 0))

    risk_map = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    curated['risk_label'] = curated['risk_level'].map(lambda r: risk_map.get(r, 0))

    total_clean = len(curated)
    mal_count = int(curated['binary_label'].sum())
    ben_count = total_clean - mal_count
    audit_stats["total_gold_clean_records"] = total_clean
    audit_stats["gold_clean_attacks"] = mal_count
    audit_stats["gold_clean_benign"] = ben_count

    print(f"\n[5/6] Total Verified Gold-Standard Prompts: {total_clean:,}")
    print(f"  - Confirmed Attacks (Class 1) : {mal_count:,} ({mal_count/total_clean*100:.1f}%)")
    print(f"  - Confirmed Safe (Class 0)    : {ben_count:,} ({ben_count/total_clean*100:.1f}%)")

    # 7. Stratified 80% Train, 20% Test Split
    print("\n[6/6] Generating Stratified 80/20 Train and Test splits...")
    np.random.seed(42)
    # Stratify by binary_label
    mal_df = curated[curated['binary_label'] == 1].sample(frac=1.0, random_state=42)
    ben_df = curated[curated['binary_label'] == 0].sample(frac=1.0, random_state=42)

    n_train_mal = int(len(mal_df) * 0.80)
    n_train_ben = int(len(ben_df) * 0.80)

    train_df = pd.concat([mal_df.iloc[:n_train_mal], ben_df.iloc[:n_train_ben]]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat([mal_df.iloc[n_train_mal:], ben_df.iloc[n_train_ben:]]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    train_path = os.path.join(out_dir, "train_gold.parquet")
    test_path = os.path.join(out_dir, "test_gold.parquet")
    report_path = os.path.join(out_dir, "cleaning_report.json")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    with open(report_path, "w") as f:
        json.dump(audit_stats, f, indent=2)

    print(f"\nSaved Train split ({len(train_df):,} rows) to: {train_path}")
    print(f"Saved Test split  ({len(test_df):,} rows) to: {test_path}")
    print(f"Saved Cleaning Report to: {report_path}")

    print("\n" + "=" * 70)
    print("SAMPLE VERIFICATION: 5 REAL ATTACKS vs 5 REAL SAFE PROMPTS")
    print("=" * 70)
    print("\n--- 5 Confirmed Malicious Attacks ---")
    for p in test_df[test_df['binary_label'] == 1]['prompt_text'].sample(5, random_state=42):
        print(f"  [ATTACK] {repr(p[:95])}")

    print("\n--- 5 Confirmed Safe Benign Queries ---")
    for p in test_df[test_df['binary_label'] == 0]['prompt_text'].sample(5, random_state=42):
        print(f"  [SAFE]   {repr(p[:95])}")

    print("=" * 70)
    print("DATASET SANITIZATION COMPLETE SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_gold_sanitization()
