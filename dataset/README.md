# Guardrailer Dataset v1

A high-quality, cleaned dataset for training and evaluating **prompt injection detection systems**. This dataset contains 722,842 labeled prompts spanning benign and malicious attack categories including jailbreaks, direct injections, and system prompt extraction attempts.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total Samples | 722,842 |
| Malicious | 433,066 (59.9%) |
| Benign | 289,776 (40.1%) |
| Columns | 8 |
| File Format | Parquet |

## Attack Categories

| Category | Count | % of Total |
|----------|-------|------------|
| `benign_control` | 289,776 | 40.1% |
| `jailbreak` | 267,091 | 36.9% |
| `direct_injection` | 92,206 | 12.8% |
| `system_prompt_extraction` | 73,769 | 10.2% |

## Risk Levels

| Level | Count | % |
|-------|-------|---|
| none | 289,776 | 40.1% |
| low | 249,949 | 34.6% |
| high | 174,380 | 24.1% |
| medium | 6,987 | 1.0% |
| critical | 1,750 | 0.2% |

## Data Sources

| Source | Count | % |
|--------|-------|---|
| llm_jailbreak_injection | 418,006 | 57.8% |
| HF:ahsanayub/malicious-prompts | 155,035 | 21.4% |
| HF:gabrielchua/system-prompt-leakage | 71,008 | 9.8% |
| HF:Lakera/mosscap_prompt_injection | 48,338 | 6.7% |
| HF:quickium/prompt-security-v0 | 22,344 | 3.1% |
| HF:aurora-m/adversarial-prompts | 8,111 | 1.1% |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique sample identifier |
| `prompt_text` | string | The prompt text content |
| `is_malicious` | bool | `True` if malicious, `False` if benign |
| `attack_category` | string | Category: `jailbreak`, `direct_injection`, `system_prompt_extraction`, or `benign_control` |
| `attack_technique` | string | Specific technique (e.g., `pattern_matching`, `extraction`, `virtualization_roleplay`, `base64_encoding`) |
| `risk_level` | string | Risk assessment: `critical`, `high`, `medium`, `low`, or `none` |
| `source_dataset` | string | Original dataset source |
| `length_norm` | float64 | Min-max normalized text length [0, 1] |

## Data Quality

This dataset has been cleaned with the following steps:

1. **Deduplication**: Removed 210,138 duplicate prompt texts (21% of original)
2. **Label reconciliation**: Fixed 114,400 rows where benign samples had incorrect attack categories
3. **Dead column removal**: Dropped `uniqueness`, `cross_encoder_score`, `neighbor_ids`, `cluster_id` (constant/empty values)
4. **Length normalization**: Recomputed `length_norm` using min-max scaling
5. **Extreme length filtering**: Removed texts <5 chars or >20K chars
6. **Jailbreak reduction**: Downsampled jailbreak category by 20% to reduce dominance

### Validation Results

- No missing values
- No duplicate texts
- All IDs unique
- Label-category consistency: 100%
- Text length bounds: 5–19,669 chars

## Usage

```python
import pandas as pd

df = pd.read_parquet('guardrailer_dataset_v1.parquet')

# Basic stats
print(f"Total: {len(df):,}")
print(f"Malicious: {df['is_malicious'].sum():,}")

# Filter by category
jailbreaks = df[df['attack_category'] == 'jailbreak']
injections = df[df['attack_category'] == 'direct_injection']

# Train/test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify=df['attack_category'], random_state=42)
```

## Citation

If you use this dataset, please cite:

```
Guardrailer Dataset v1 - Prompt Injection Detection Corpus
722,842 labeled prompts for training security classifiers
Published: 2026
```

## License

Research use. See project repository for details.
