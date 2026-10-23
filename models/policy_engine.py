#!/usr/bin/env python3
"""
Step 5: Policy-Based Defense Engine for HLS-Guard 2.0 (Objective 7)
Translates deep learning multi-task predictions (Malicious Probability,
Attack Category, Risk Severity) into deterministic operational decisions:
- ALLOW : Verified safe request -> Forwarded directly to target LLM.
- FLAG  : Borderline uncertainty or medium risk -> Logged / Secondary inspection.
- BLOCK : Confirmed attack or high/critical risk -> Rejected immediately.
"""

import os
import sys
import time
import torch
import numpy as np
from transformers import AutoTokenizer

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hls_guard import GatedHLSGuard
from models.dataset_loader import (
    extract_pattern_features,
    PatternScaler,
    FEATURE_NAMES
)

CATEGORY_MAP = {
    0: "benign_control",
    1: "direct_injection",
    2: "jailbreak",
    3: "system_prompt_extraction"
}

RISK_MAP = {
    0: "none",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical"
}

class PolicyDefenseEngine:
    """
    Operational Security Gateway for LLM Applications.
    Combines Tri-Head Gated HLS-Guard with deterministic policy rules.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_hls_guard.pt",
        scaler_path: str = "checkpoints/pattern_scaler.pkl",
        model_name: str = "distilroberta-base",
        allow_threshold: float = 0.20,
        block_threshold: float = 0.70,
        device: str = "auto"
    ):
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold

        self.device = torch.device(
            device if device != "auto" else (
                "cuda" if torch.cuda.is_available() else (
                    "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        )

        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 2. Load Pattern Scaler
        if not os.path.exists(scaler_path):
            scaler_path = "models/pattern_scaler.pkl"
        self.scaler = PatternScaler(scaler_path=scaler_path)

        # 3. Load Model
        self.model = GatedHLSGuard(
            model_name=model_name,
            num_pattern_features=14,
            num_categories=4,
            num_risk_levels=5,
            fusion_dim=256,
            dropout=0.0
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[PolicyEngine] Loaded model checkpoint from: {checkpoint_path}")
        else:
            print(f"[PolicyEngine] Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        self.model.eval()

    def inspect_prompt(self, prompt_text: str) -> dict:
        """
        Inspects an input prompt and applies policy rules to return ALLOW, FLAG, or BLOCK.
        """
        t0 = time.perf_counter()
        clean_text = str(prompt_text).strip()

        # 1. Extract and scale pattern features
        feat_dict = extract_pattern_features(clean_text)
        feat_row = np.array([[feat_dict[k] for k in FEATURE_NAMES]], dtype=np.float32)
        scaled_feat = self.scaler.load_and_transform(feat_row)
        pat_tensor = torch.tensor(scaled_feat, dtype=torch.float32, device=self.device)

        # 2. Tokenize with head-tail preservation
        words = clean_text.split()
        if len(words) > 240:
            head = " ".join(words[:120])
            tail = " ".join(words[-120:])
            formatted_text = f"{head} ... [MID_TEXT_OMITTED] ... {tail}"
        else:
            formatted_text = clean_text

        encoded = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, pat_tensor)

            bin_probs = torch.softmax(outputs["binary_logits"], dim=-1)[0].cpu().numpy()
            cat_probs = torch.softmax(outputs["category_logits"], dim=-1)[0].cpu().numpy()
            risk_probs = torch.softmax(outputs["risk_logits"], dim=-1)[0].cpu().numpy()
            gate_w = outputs["gate_weights"][0].cpu().numpy()

        p_malicious = float(bin_probs[1])
        cat_id = int(np.argmax(cat_probs))
        cat_name = CATEGORY_MAP[cat_id]
        cat_conf = float(cat_probs[cat_id])

        risk_id = int(np.argmax(risk_probs))
        risk_name = RISK_MAP[risk_id]
        risk_conf = float(risk_probs[risk_id])

        # Dynamic gating interpretation
        mean_gate = float(np.mean(gate_w))
        dominant_branch = "Semantic Context" if mean_gate >= 0.50 else "Pattern Scanner"

        # 4. Policy Decision Rules (Objective 7)
        # Check hard trigger keywords as defense-in-depth
        has_override_flag = feat_dict.get("override_instruction", 0) == 1
        has_extraction_flag = feat_dict.get("system_prompt_extraction", 0) == 1
        has_dan_flag = feat_dict.get("role_manipulation", 0) == 1

        # Rule 1: High Severity BLOCK
        if (
            p_malicious >= self.block_threshold or
            risk_name in ["critical", "high"] or
            (has_override_flag and has_extraction_flag) or
            has_dan_flag
        ):
            decision = "BLOCK"
            reason = (
                f"Malicious prompt injection detected. Attack category: [{cat_name.upper()}], "
                f"Risk tier: [{risk_name.upper()}] (Malicious Probability: {p_malicious*100:.1f}%)."
            )

        # Rule 2: Moderate Uncertainty FLAG
        elif (
            (self.allow_threshold <= p_malicious < self.block_threshold) or
            risk_name == "medium" or
            has_override_flag or
            has_extraction_flag
        ):
            decision = "FLAG"
            reason = (
                f"Borderline / suspicious prompt flagged for secondary review. "
                f"Indicated category: [{cat_name}], Risk: [{risk_name}] "
                f"(Probability: {p_malicious*100:.1f}%)."
            )

        # Rule 3: Low Risk Verified ALLOW
        else:
            decision = "ALLOW"
            reason = f"Verified benign query. Low threat risk [{risk_name}] (Confidence: {(1.0-p_malicious)*100:.1f}%)."

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "decision": decision,
            "prompt": clean_text,
            "malicious_probability": round(p_malicious, 4),
            "safety_score": round(1.0 - p_malicious, 4),
            "predicted_category": cat_name,
            "category_confidence": round(cat_conf, 4),
            "risk_level": risk_name,
            "risk_confidence": round(risk_conf, 4),
            "action_reason": reason,
            "latency_ms": round(latency_ms, 2),
            "gate_analysis": {
                "mean_gate_weight": round(mean_gate, 4),
                "dominant_branch": dominant_branch
            }
        }
