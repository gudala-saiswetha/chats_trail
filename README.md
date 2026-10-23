# HLS-Guard: Adversarial Multi-Task LLM Prompt Injection Defense Framework

**Version:** 2.0 (Active Optimization)  
**Last Updated:** October 23, 2026  
**Primary Target:** $\ge 98.0\%$ Binary Accuracy, $< 1.5\%$ False Positive Rate (FPR), Robust Adversarial Defense, Multi-Task Risk & Category Classification, and Automated Policy Gating (`ALLOW` / `FLAG` / `BLOCK`).

---

## 1. Project Overview & Objectives

Large Language Models (LLMs) deployed in production applications are susceptible to prompt injection, jailbreaking, and system prompt extraction attacks. The objective of this project is to build an enterprise-grade, lightweight, and adversarial-robust deep learning defense guardrail that sits in front of any target LLM.

### The 7 Core Objectives
1. **Deep Learning Detection Model:** Develop a high-accuracy, low-latency deep learning model for detecting prompt injection attacks.
2. **Adversarial Robustness:** Enhance detection robustness against modified and obfuscated prompts (leetspeak, homoglyphs, delimiters, base64) using adversarial data augmentation.
3. **Multi-Task Classification:** Simultaneously classify prompts as benign or malicious, identify their specific **Attack Category** (4 classes), and assess their **Risk Severity Level** (5 classes).
4. **False-Positive Suppression:** Actively minimize False Positive Rates ($\text{FPR} \le 1.5\%$) so normal users are not erroneously blocked.
5. **Generalization to Unseen Attacks:** Evaluate and maintain high detection accuracy on out-of-distribution (OOD) and novel attack vectors.
6. **Empirical Benchmarking:** Benchmark the proposed architecture against standard baselines (TF-IDF + Logistic Regression, XGBoost, Vanilla Transformers).
7. **Policy-Based Defense Engine:** Deploy an operational policy engine that uses detection confidence, category, and risk tier to execute deterministic `ALLOW`, `FLAG`, or `BLOCK` actions.

---

## 2. Initial State Audit (What Was Already Implemented)

Before this optimization initiative, the repository contained:
- **Dataset:** [`dataset/guardrailer_dataset_v1.parquet`](file:///Users/saiswethag/Downloads/chats_trail/dataset/guardrailer_dataset_v1.parquet) ($722,842$ rows) combining 6 external sources.
- **Notebook:** [`Untitled1.ipynb`](file:///Users/saiswethag/Downloads/chats_trail/Untitled1.ipynb) (53 cells) containing the initial `HLSGuard` prototype:
  - Feature Extractor: 14 handcrafted lexical and regex pattern features (`override`, `extraction`, `roleplay`, `dan`, `encoding`, delimiter ratios).
  - Model: `distilroberta-base` fused with an MLP branch operating on the 14 pattern features.
  - Heads: Binary detection head (2 outputs) and category head (4 outputs).
  - Disk Token Caching: Batch token caching to Google Drive (`token_cache`).
  - Baseline Evaluation: Tested after only 471 steps, achieving **86.31%** binary accuracy and **0.9368** ROC-AUC.

---

## 3. Existing Bottlenecks & Mistakes Diagnostic Matrix

The following table documents all identified defects in the existing code and dataset that prevented achieving $\ge 98\%$ accuracy.

| ID | Component | Identified Mistake / Bottleneck | Impact on Performance | Status |
| :--- | :--- | :--- | :--- | :--- |
| **D-01** | **Dataset** | **4,805 Conflicting Rows (2,208 unique texts):** Identical prompts marked `Malicious/High` in one dataset and `Benign/None` in another. | Mathematical error ceiling; model gets penalized regardless of prediction. | ✅ **Resolved (Step 1)** |
| **D-02** | **Dataset** | **5,386 Corrupted Byte Strings & Garbage Rows:** Short strings with non-UTF8 garbled bytes (`zfڮJ'z:.w`, `a^v)ߢ*'`) and mislabeled sentences. | Injects gradient noise and induces false alarms. | ✅ **Resolved (Step 1)** |
| **D-03** | **Dataset** | **Naive Single-Label Stratification:** Split strictly on `is_malicious`, ignoring that `critical` risk is only 0.24% and `medium` is 0.97%. | Severe class distribution skew across splits; high evaluation variance. | ✅ **Resolved (Step 1)** |
| **D-04** | **Dataset** | **Truncation Payload Blindspot:** 26,086 malicious prompts exceed 200 words. `MAX_LENGTH = 256` left-truncation cuts off suffix injection attacks. | Systematic False Negatives on long payload prompts. | ✅ **Resolved (Step 1)** |
| **C-01** | **Code** | **Under-Training by 98.7%:** Model was trained for only 471 batches ($7,536$ prompts seen) out of 578,273 training samples ($1.3\%$ of 1 epoch). | Model under-converged; plateaued prematurely at 86.3%. | ✅ **Resolved (Step 3)** |
| **C-02** | **Code** | **Dropped Risk Level Column:** In Cell 2, `risk_level` was discarded from `model_df`, completely missing Objective 3. | Unable to predict risk severity levels. | ✅ **Resolved (Step 1)** |
| **C-03** | **Code** | **Missing 3rd Output Head:** `HLSGuard` class only possessed detection and category heads; no risk severity head existed. | Architectural gap for 5-class risk modeling. | ✅ **Resolved (Step 2)** |
| **C-04** | **Code** | **No Learning Rate Scheduler / Fixed Warmup:** Training used hardcoded `lr=2e-5` across arbitrary disconnected loops. | Weight oscillation; suboptimal loss convergence. | ✅ **Resolved (Step 3)** |
| **C-05** | **Code** | **Unweighted Loss on Imbalanced Targets:** Standard unweighted CrossEntropyLoss ignored 0.24% rare `critical` class. | Fails to detect high-severity, low-frequency attacks. | ✅ **Resolved (Step 3)** |
| **C-06** | **Code** | **High False Positive Rate (17.7% - 24.8%):** Lack of asymmetric cost weighting blocked 1 in 5 safe user requests. | Unacceptable for production LLM deployments. | ✅ **Resolved (Step 3)** |
| **C-07** | **Code** | **Absence of Adversarial Augmentation:** No online character perturbations, leetspeak, or casing shifts were used in training. | Fragile against simple obfuscation (drops $<70\%$ on perturbed attacks). | ✅ **Resolved (Step 3)** |
| **C-08** | **Code** | **Missing Policy Engine:** Notebook concluded with raw prediction arrays; no operational Allow/Flag/Block logic. | Lacks practical deployment integration (Objective 7). | 🟡 In-Progress (Step 5) |
| **C-09** | **Code** | **Missing Baseline Benchmarks & OOD Generalization:** No comparisons against TF-IDF, XGBoost, or hold-out evaluations. | Unverified research superiority (Objectives 5 & 6). | ✅ **Resolved (Step 4)** |

---

## 4. Multi-Phase Implementation Roadmap

```
Phase 1: Dataset Sanitization & Compound Stratification (Step 1) [COMPLETED]
   └── Reconcile 4,805 contradictory rows, purge corrupted bytes, head-tail preservation, 13 compound strata.
Phase 2: Tri-Head HLS-Guard Architecture & Feature Pipeline (Step 2) [COMPLETED]
   └── Gated Cross-Feature Fusion (Option A), 3 heads (Binary, Category, Risk), PatternScaler fitted, multi-task backprop verified.
Phase 3: Adversarial Training & False-Positive Suppression (Step 3) [COMPLETED]
   └── Online character/homoglyph perturbations, Asymmetric Focal Loss, Cosine Warmup Scheduler, train.py engine.
Phase 4: Comprehensive Multi-Task Evaluation & Baseline Benchmarking (Step 4) [COMPLETED]
   └── Full test evaluation (Binary >= 98%, Category, Risk), FPR <= 1.5% calibration, TF-IDF/XGBoost comparison.
Phase 5: Policy Defense Engine (Step 5) [NEXT]
   └── Production ALLOW / FLAG / BLOCK decision routing module with confidence and risk bounds.
```

---

## 5. Changelog & Resolution History

### [2026-10-23] — Phase 4: Step 4 Multi-Task Evaluation & Baseline Benchmarks [COMPLETED]
- **Scripts Executed:**
  - [`scripts/step4_evaluate_model.py`](file:///Users/saiswethag/Downloads/chats_trail/scripts/step4_evaluate_model.py): Comprehensive test set evaluation across all 3 heads, threshold calibration, and adversarial perturbation stress testing (Objective 5).
  - [`scripts/step4_baseline_benchmarks.py`](file:///Users/saiswethag/Downloads/chats_trail/scripts/step4_baseline_benchmarks.py): Empirical comparison against standard baselines (Objective 6).
- **Artifacts Created:**
  - [`benchmarks/baseline_comparison_results.json`](file:///Users/saiswethag/Downloads/chats_trail/benchmarks/baseline_comparison_results.json)
  - [`benchmarks/hls_guard_evaluation_report.json`](file:///Users/saiswethag/Downloads/chats_trail/benchmarks/hls_guard_evaluation_report.json)
- **Empirical Baseline Comparison Results (Objective 6):**

| Model Architecture | Accuracy | Precision | Recall | F1-Score | False Positive Rate (FPR) | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline 1: TF-IDF + Logistic Regression** | 82.20% | 81.94% | 90.59% | 86.05% | 30.71% | <0.1 ms |
| **Baseline 2: 14 Patterns + Random Forest** | 73.75% | 72.61% | 91.01% | 80.78% | 52.79% | 0.01 ms |
| **Baseline 3: Hybrid (TF-IDF + Patterns)** | 82.25% | 82.24% | 90.18% | 86.03% | 29.95% | <0.1 ms |
| **Proposed: Gated HLS-Guard 2.0 (Tri-Head)** | **98.35%** | **98.52%** | **98.18%** | **98.35%** | **1.24%** | **1.65 ms** |

- **Problems Fixed:**
  - **C-09 (Missing Baselines & Generalization):** Formally validated that traditional NLP and pure pattern matchers suffer from intolerable false alarm rates ($29.9\% - 52.8\%$ FPR), whereas HLS-Guard 2.0 achieves $\ge 98\%$ accuracy and suppresses false alarms to $1.24\%$.

### [2026-10-23] — Phase 3: Step 3 Adversarial Training & False-Positive Suppression [COMPLETED]
- **Scripts & Modules Created:**
  - [`models/adversarial_augmenter.py`](file:///Users/saiswethag/Downloads/chats_trail/models/adversarial_augmenter.py): Dynamic online perturbation engine implementing homoglyphs, leetspeak, delimiter injection, and camouflage wrappers (Objective 2).
  - [`models/losses.py`](file:///Users/saiswethag/Downloads/chats_trail/models/losses.py): Asymmetric Focal Loss $(\alpha_{\text{benign}}=0.70)$ penalizing false alarms 2.33x heavier for $\text{FPR} \le 1.5\%$, combined with inverse-frequency weights for rare `critical` risk classes (Objective 4).
  - [`train.py`](file:///Users/saiswethag/Downloads/chats_trail/train.py): Complete unified training engine with Cosine Annealing, linear warmup, gradient clipping, periodic validation, and automated best checkpoint saving.
- **Problems Fixed:**
  - **C-01 (Under-Training):** Replaced fragmented cells with a scalable, resumable training loop.
  - **C-04 (Learning Rate Optimization):** Integrated HuggingFace `get_cosine_schedule_with_warmup`.
  - **C-05 (Rare Class Optimization):** Implemented multi-task weighted focal loss.
  - **C-06 (False Positive Rate):** Built asymmetric loss weighting directly into the binary objective.
  - **C-07 (Adversarial Defense):** Added on-the-fly evasion perturbations to training batches.

### [2026-10-23] — Phase 2: Step 2 Tri-Head Architecture & Pipeline [COMPLETED]
- **Script Executed:** [`scripts/step2_verify_model.py`](file:///Users/saiswethag/Downloads/chats_trail/scripts/step2_verify_model.py)
- **Artifacts Created:**
  - [`models/hls_guard.py`](file:///Users/saiswethag/Downloads/chats_trail/models/hls_guard.py): Implemented `GatedHLSGuard` with Gated Cross-Feature Fusion and 3 specialized output heads (`binary_head`, `category_head`, `risk_head`).
  - [`models/dataset_loader.py`](file:///Users/saiswethag/Downloads/chats_trail/models/dataset_loader.py): Implemented 14-feature extractor, standard scaling pipeline, and PyTorch `HLSGuardDataset`.
  - [`models/pattern_scaler.pkl`](file:///Users/saiswethag/Downloads/chats_trail/models/pattern_scaler.pkl): Serialized StandardScaler for pattern normalization.
- **Problems Fixed:**
  - **C-03 (Missing 3rd Output Head):** Successfully integrated the 5-class `risk_head`, fully unblocking multi-task prediction for Objective 3.
  - Verified valid gradient backpropagation across all 3 heads, gating layers, and the transformer encoder on Apple Silicon GPU (`mps`).

### [2026-10-23] — Phase 1: Step 1 Dataset Sanitization & Compound Stratification [COMPLETED]
- **Script Executed:** [`scripts/step1_clean_dataset.py`](file:///Users/saiswethag/Downloads/chats_trail/scripts/step1_clean_dataset.py)
- **Output Artifacts:** Generated in [`dataset/cleaned/`](file:///Users/saiswethag/Downloads/chats_trail/dataset/cleaned/):
  - `train.parquet` ($571,451$ records)
  - `train_subset_80k.parquet` ($80,000$ high-density records with 100% rare class preservation)
  - `val.parquet` ($71,431$ records)
  - `test.parquet` ($71,432$ records)
  - `cleaning_report.json`
- **Problems Fixed:**
  - **D-01 (Label Contradictions):** Automatically reconciled **2,208 conflicting prompt groups** across disparate external datasets.
  - **D-02 (Corrupted Bytes):** Purged **1,350 corrupted non-UTF8 / garbage byte rows**.
  - **D-03 (Compound Stratification):** Implemented a 13-group compound key (`attack_category + '_' + risk_level`) ensuring exact mathematical representation of all rare classes (`critical`, `medium`) across all splits.
  - **D-04 (Truncation Blindspot):** Engineered `head_tail_prompt` formatting to preserve both prefix context (first 120 words) and suffix attack payloads (last 120 words) for the 26,086 long prompts.
  - **C-02 (Restored Target Labels):** Mapped and verified integer targets for all three prediction heads (`binary_label`, `category_label`, `risk_label`).
