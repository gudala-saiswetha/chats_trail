#!/usr/bin/env python3
"""
Asymmetric Multi-Task Focal Loss Functions for HLS-Guard 2.0
Designed to strictly suppress False Positives (<1.5% FPR) and compensate
for extreme class imbalances in attack categories and risk levels (Objective 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for Binary Classification.
    Penalizes False Positives (blocking a benign user) significantly heavier
    than False Negatives to maintain an ultra-low False Positive Rate (FPR <= 1.5%).
    """
    def __init__(self, alpha_benign: float = 0.70, gamma: float = 2.0):
        super().__init__()
        # alpha_benign weights errors on class 0 (benign) heavier
        self.alpha = torch.tensor([alpha_benign, 1.0 - alpha_benign])
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(logits.device)
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = alpha.gather(0, targets)

        # Focal weighting factor
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        loss = focal_weight * F.cross_entropy(logits, targets, reduction="none")
        return loss.mean()


class WeightedMultiClassFocalLoss(nn.Module):
    """
    Focal Loss with class inverse frequency weighting for multi-class targets.
    Forces the network to attend to rare classes like 'critical' risk (0.24%).
    """
    def __init__(self, class_weights: list, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        w_t = self.weights[targets]

        focal_weight = w_t * ((1.0 - pt) ** self.gamma)
        loss = focal_weight * F.cross_entropy(logits, targets, reduction="none")
        return loss.mean()


class MultiTaskHLSGuardLoss(nn.Module):
    """
    Combined Multi-Task Loss:
    L_total = w_bin * L_binary + w_cat * L_category + w_risk * L_risk
    """
    def __init__(
        self,
        alpha_benign: float = 0.70,
        gamma: float = 2.0,
        w_binary: float = 1.0,
        w_category: float = 0.5,
        w_risk: float = 0.5,
    ):
        super().__init__()
        self.w_binary = w_binary
        self.w_category = w_category
        self.w_risk = w_risk

        # 1. Asymmetric loss on binary head (FPR suppression)
        self.binary_loss_fn = AsymmetricFocalLoss(alpha_benign=alpha_benign, gamma=gamma)

        # 2. Category loss: boosted for direct_injection (index 1) which was hardest class
        # Classes: 0: benign_control, 1: direct_injection, 2: jailbreak, 3: system_prompt_extraction
        self.category_loss_fn = WeightedMultiClassFocalLoss(
            class_weights=[1.0, 2.5, 1.2, 1.8],
            gamma=gamma
        )

        # 3. Risk loss: heavily boosted for critical (index 4 = 1,750 samples) and medium (index 2 = 6,987)
        # Classes: 0: none, 1: low, 2: medium, 3: high, 4: critical
        self.risk_loss_fn = WeightedMultiClassFocalLoss(
            class_weights=[1.0, 1.2, 5.0, 1.5, 10.0],
            gamma=gamma
        )

    def forward(
        self,
        binary_logits: torch.Tensor,
        category_logits: torch.Tensor,
        risk_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        category_targets: torch.Tensor,
        risk_targets: torch.Tensor
    ):
        loss_b = self.binary_loss_fn(binary_logits, binary_targets)
        loss_c = self.category_loss_fn(category_logits, category_targets)
        loss_r = self.risk_loss_fn(risk_logits, risk_targets)

        total_loss = (self.w_binary * loss_b) + (self.w_category * loss_c) + (self.w_risk * loss_r)

        return {
            "total_loss": total_loss,
            "binary_loss": loss_b,
            "category_loss": loss_c,
            "risk_loss": loss_r
        }
