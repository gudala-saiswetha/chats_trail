#!/usr/bin/env python3
"""
Feature Extraction, Scaling, and PyTorch DataLoader for HLS-Guard 2.0
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

PATTERN_GROUPS = {
    "override_instruction": [
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\bdisregard (all )?(previous|prior|above) instructions\b",
        r"\boverride (the )?(system|previous) instructions\b",
        r"\bforget (all )?(previous|prior) instructions\b",
    ],
    "system_prompt_extraction": [
        r"\breveal (the )?(system prompt|hidden prompt)\b",
        r"\bshow (me )?(the )?(system prompt|hidden instructions)\b",
        r"\brepeat (the )?(system prompt|instructions)\b",
        r"\bprint (your )?(system prompt|initial instructions)\b",
    ],
    "role_manipulation": [
        r"\byou are now\b",
        r"\bact as (an? )?(unrestricted|different|new)\b",
        r"\bpretend (that )?you are\b",
        r"\bdeveloper mode\b",
        r"\bdan mode\b",
    ],
    "instruction_boundary": [
        r"<\s*/?\s*(system|instruction|prompt)\s*>",
        r"\[\s*(system|instruction|prompt)\s*\]",
        r"###\s*(system|instruction|user|assistant)\b",
    ],
    "encoding_indicators": [
        r"\bbase64\b",
        r"\bdecode this\b",
        r"\bfrom hex\b",
        r"\bunhexlify\b",
        r"\batob\s*\(",
    ],
    "secrecy_or_extraction": [
        r"\bhidden instructions\b",
        r"\bconfidential prompt\b",
        r"\bprivate system message\b",
        r"\bdo not disclose\b",
    ],
}

FEATURE_NAMES = [
    "char_count",
    "word_count",
    "line_count",
    "digit_ratio",
    "uppercase_ratio",
    "special_char_ratio",
    "url_count",
    "code_fence_count",
    "override_instruction",
    "system_prompt_extraction",
    "role_manipulation",
    "instruction_boundary",
    "encoding_indicators",
    "secrecy_or_extraction",
]

def extract_pattern_features(text: str) -> dict:
    """Extracts 14 handcrafted lexical and regex features from a prompt."""
    text_str = str(text)
    lower = text_str.lower()
    length = max(len(text_str), 1)

    features = {
        "char_count": len(text_str),
        "word_count": len(text_str.split()),
        "line_count": len(text_str.splitlines()),
        "digit_ratio": sum(c.isdigit() for c in text_str) / length,
        "uppercase_ratio": sum(c.isupper() for c in text_str) / length,
        "special_char_ratio": sum(not c.isalnum() and not c.isspace() for c in text_str) / length,
        "url_count": len(re.findall(r"https?://\S+", text_str, re.I)),
        "code_fence_count": text_str.count("```"),
    }

    for group, patterns in PATTERN_GROUPS.items():
        features[group] = int(any(re.search(pat, lower, re.I) for pat in patterns))

    return features

def build_pattern_matrix(df: pd.DataFrame, text_column: str = "prompt_text") -> np.ndarray:
    """Extracts the 14 pattern features for all rows in a DataFrame."""
    rows = [extract_pattern_features(t) for t in df[text_column]]
    matrix = pd.DataFrame(rows)[FEATURE_NAMES].to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


class PatternScaler:
    """Handles standard scaling of the 14 pattern features."""
    def __init__(self, scaler_path: str = "models/pattern_scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = StandardScaler()

    def fit_and_save(self, matrix: np.ndarray):
        scaled = self.scaler.fit_transform(matrix)
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Pattern feature scaler saved to: {self.scaler_path}")
        return scaled.astype(np.float32)

    def load_and_transform(self, matrix: np.ndarray) -> np.ndarray:
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found at: {self.scaler_path}. Run fit_and_save first.")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        return self.scaler.transform(matrix).astype(np.float32)


class HLSGuardDataset(Dataset):
    """
    PyTorch Dataset providing:
    - input_ids & attention_mask (tokenized transformer representations)
    - pattern_features (14 scaled lexical floats)
    - binary_labels (0: benign, 1: malicious)
    - category_labels (0..3)
    - risk_labels (0..4)
    """
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pattern_features: np.ndarray,
        binary_labels: np.ndarray,
        category_labels: np.ndarray,
        risk_labels: np.ndarray,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.pattern_features = torch.tensor(pattern_features, dtype=torch.float32)
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        self.category_labels = torch.tensor(category_labels, dtype=torch.long)
        self.risk_labels = torch.tensor(risk_labels, dtype=torch.long)

    def __len__(self):
        return len(self.binary_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "pattern_features": self.pattern_features[idx],
            "binary_labels": self.binary_labels[idx],
            "category_labels": self.category_labels[idx],
            "risk_labels": self.risk_labels[idx],
        }

