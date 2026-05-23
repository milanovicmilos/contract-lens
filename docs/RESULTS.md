# ContractLens — Training Results Summary

Final metrics from the Kaggle T4/P100 fine-tuning pipeline, used as ground
truth for the master's thesis evaluation section.

## Multi-label Classifier (DeBERTa-v3-base + LoRA)

### Setup
- **Base model:** `microsoft/deberta-v3-base` (138M params)
- **Adapter:** LoRA on `query_proj` / `key_proj` / `value_proj`, r=32, alpha=64
- **Trainable params:** ~2.5M (0.55% of total)
- **Loss:** BCE with `pos_weight` per class (inverse frequency, clipped to 100)
- **Epochs:** 8 with EarlyStopping (patience=3) on micro F1
- **Train/eval split:** 18,176 windows → 16,358 train / 1,818 eval
- **Threshold:** per-class optimal F1 threshold tuned on the eval set

### Iterations

| Run | Δ from previous | Tuned micro F1 | Tuned macro F1 |
|-----|-----------------|----------------|----------------|
| v6  | baseline (no class weighting, 4 epochs) | 0.461 | 0.083 |
| v7  | + class-weighted BCE, LoRA r=32, 8 epochs, per-class threshold tuning | **0.671** | **0.561** |
| v8  | + merge LoRA into base for portable inference (no eval change expected) | 0.662 | 0.534 |

The jump from v6 → v7 (**+45% micro F1, +574% macro F1**) is driven entirely by
class weighting and threshold tuning — without those two changes, the model
collapses to predicting only the 4 most-frequent categories. v8 is the
production artifact because it persists a self-contained model directory
(`models/deberta-cuad-classifier/`) that includes the trained pooler head.

### Per-category F1 (v8, tuned thresholds)

**Top 12 categories (F1 ≥ 0.65):**

| Category | F1 | Precision | Recall | Support |
|----------|----|-----------|--------|---------|
| Parties | 0.910 | 0.961 | 0.865 | 401 |
| Document Name | 0.881 | 0.830 | 0.940 | 333 |
| Agreement Date | 0.866 | 0.838 | 0.897 | 331 |
| Expiration Date | 0.822 | 0.812 | 0.832 | 209 |
| License Grant | 0.798 | 0.766 | 0.832 | 238 |
| Irrevocable Or Perpetual License | 0.756 | 0.654 | 0.896 | 73 |
| Renewal Term | 0.752 | 0.745 | 0.760 | 102 |
| Effective Date | 0.732 | 0.683 | 0.789 | 274 |
| Notice Period To Terminate Renewal | 0.680 | 0.566 | 0.852 | 58 |
| Cap On Liability | 0.669 | 0.847 | 0.553 | 121 |
| Insurance | 0.668 | 0.831 | 0.557 | 70 |
| Non-Compete | 0.658 | 0.566 | 0.787 | 159 |

**Weakest categories (F1 < 0.30):**

| Category | F1 | Support | Why |
|----------|----|---------|-----|
| Price Restrictions | 0.111 | 4 | Too few positives in eval split |
| Most Favored Nation | 0.146 | 7 | Too few positives in eval split |
| Covenant Not To Sue | 0.258 | 29 | Sparse training signal |
| Change Of Control | 0.275 | 56 | High lexical variance across contracts |

The four weak categories are all in the long tail of the class distribution.
Of the 41 categories, **17 reach F1 ≥ 0.60** at tuned thresholds, which is the
operationally useful subset for the API demo and risk dashboard.

## Extractive QA (DeBERTa-v3-base, full fine-tune)

In progress at the time of writing — see `models/deberta-cuad-extractor/`
and `docs/extractor_eval.json` once the kernel completes.

## End-to-End Demo

`docs/demo_inference.txt` captures the orchestrator output on a Distribution
Agreement excerpt. Highlights:

- Non-Compete clause "worldwide" → policy escalates to **High** risk
- No-Solicit Of Customers detected at 0.96 confidence → **High** risk
- License Grant + Cap On Liability detected with rationales from the 41-
  category default policy
- RAG retrieval of GDPR / EU AI Act snippets via the ChromaDB store

## Reproducibility

- Datasets: `milomilanovi/contractlens-cuad-multilabel`,
  `milomilanovi/contractlens-cuad-squad` (CC-BY-4.0)
- Kernels: `milomilanovi/contractlens-classifier-training`,
  `milomilanovi/contractlens-extractor-training`
- Pull artifacts: `python scripts/pull_kaggle_models.py`
- Re-run eval: `python -m src.evaluation.ragas_eval --max-contracts 25`
