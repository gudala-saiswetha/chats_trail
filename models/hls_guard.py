#!/usr/bin/env python3
"""
HLS-Guard 2.0: Gated Cross-Feature Fusion Architecture
Multi-Task Prompt Injection, Attack Category, and Risk Severity Defense Model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class GatedFusion(nn.Module):
    """
    Gated Cross-Feature Fusion Mechanism:
    Dynamically balances semantic contextual embeddings from the Transformer
    and lexical red-flag indicators from the handcrafted pattern branch.
    """
    def __init__(self, semantic_dim: int, pattern_dim: int, fusion_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj_semantic = nn.Sequential(
            nn.Linear(semantic_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        self.proj_pattern = nn.Sequential(
            nn.Linear(pattern_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        # Learnable gating vector
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, semantic_vec: torch.Tensor, pattern_vec: torch.Tensor):
        h_sem = self.proj_semantic(semantic_vec)
        h_pat = self.proj_pattern(pattern_vec)

        # Gate determines dynamic weighting between semantic context and keyword patterns
        concat = torch.cat([h_sem, h_pat], dim=-1)
        g = self.gate(concat)

        fused = (g * h_sem) + ((1.0 - g) * h_pat)
        fused = self.out_norm(fused)
        fused = self.dropout(fused)
        return fused, g


class GatedHLSGuard(nn.Module):
    """
    Tri-Head HLS-Guard 2.0:
    - Branch A: Pretrained Transformer Encoder (DistilRoBERTa / DeBERTa-v3)
    - Branch B: 14 Handcrafted Lexical/Regex Pattern MLP
    - Fusion: Gated Cross-Feature Fusion
    - Heads:
        1. Binary Detection Head: 2 classes (Benign vs Malicious)
        2. Attack Category Head: 4 classes (Benign, Direct Injection, Jailbreak, Extraction)
        3. Risk Severity Head: 5 classes (None, Low, Medium, High, Critical)
    """
    def __init__(
        self,
        model_name: str = "distilroberta-base",
        num_pattern_features: int = 14,
        num_categories: int = 4,
        num_risk_levels: int = 5,
        fusion_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.model_name = model_name
        self.num_pattern_features = num_pattern_features
        self.num_categories = num_categories
        self.num_risk_levels = num_risk_levels

        # Branch A: Transformer Semantic Encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.semantic_encoder = AutoModel.from_pretrained(model_name, config=self.config)
        semantic_dim = self.config.hidden_size

        # Branch B: Handcrafted Pattern Feature MLP
        self.pattern_branch = nn.Sequential(
            nn.Linear(num_pattern_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # Gated Cross-Feature Fusion Layer
        self.fusion = GatedFusion(
            semantic_dim=semantic_dim,
            pattern_dim=64,
            fusion_dim=fusion_dim,
            dropout=dropout
        )

        # Output Heads (All 3 Targets)
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

        self.category_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_categories)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_risk_levels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pattern_features: torch.Tensor
    ):
        # 1. Semantic contextual representation
        encoder_out = self.semantic_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] representation (first token)
        cls_vec = encoder_out.last_hidden_state[:, 0, :]

        # 2. Pattern representation
        pat_vec = self.pattern_branch(pattern_features)

        # 3. Gated fusion
        fused_repr, gate_weights = self.fusion(cls_vec, pat_vec)

        # 4. Multi-head logits
        binary_logits = self.binary_head(fused_repr)
        category_logits = self.category_head(fused_repr)
        risk_logits = self.risk_head(fused_repr)

        return {
            "binary_logits": binary_logits,
            "category_logits": category_logits,
            "risk_logits": risk_logits,
            "gate_weights": gate_weights,
            "fused_representation": fused_repr
        }

