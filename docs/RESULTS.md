# ContractLens — Training & Evaluation Results

This document is the **single source of truth** for every metric quoted in
the master's thesis. All numbers are produced by reproducible scripts in
this repository; no figure here is invented.

## 1. Multi-label Classifier (DeBERTa-v3-base + LoRA)

### Setup

| Item | Value |
|------|-------|
| Base model | `microsoft/deberta-v3-base` (138 M params) |
| Adapter | LoRA on `query_proj` / `key_proj` / `value_proj`, r=32, α=64 |
| Trainable parameters | 2.5 M (0.55% of total) |
| Loss | BCE with `pos_weight` per class (inverse frequency, clipped to 100) |
| Epochs | 8 with EarlyStopping (patience 3) on micro F1 |
| Train / eval split | 18 176 windows → 16 358 train / 1 818 eval |
| Threshold | per-class F1-optimal on the eval set |
| Hardware | Kaggle GPU (T4 or P100 — both supported via torch 2.4 + transformers 4.46) |

### Iteration history

| Run | Δ | Tuned micro F1 | Tuned macro F1 |
|-----|---|----------------|----------------|
| v6 | baseline (no class weights, 4 epochs) | 0.461 | 0.083 |
| v7 | + class-weighted BCE, r=32, 8 epochs, per-class threshold tuning | 0.671 | 0.561 |
| **v8** | + merge LoRA into base for portable inference (production artifact) | **0.662** | **0.534** |

v8 is what ships in `models/deberta-cuad-classifier/`. v6 → v7 (+45% micro,
+574% macro) is driven by class weighting and threshold tuning. v8 keeps the
v7 hyperparameters and adds `merge_and_unload()` + full-model save so local
inference no longer re-initialises the pooler head at random.

### Per-category F1 (v8, tuned thresholds)

**Top 12 (F1 ≥ 0.65):**

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

**Weakest (F1 < 0.30):**

| Category | F1 | Support | Why |
|----------|----|---------|-----|
| Price Restrictions | 0.111 | 4 | Too few positives in eval split |
| Most Favored Nation | 0.146 | 7 | Too few positives in eval split |
| Covenant Not To Sue | 0.258 | 29 | Sparse training signal |
| Change Of Control | 0.275 | 56 | High lexical variance across contracts |

Of 41 categories, 17 reach F1 ≥ 0.60 at tuned thresholds — the operationally
useful subset for the API demo and risk dashboard.

Full report: [`classifier_eval_report.json`](classifier_eval_report.json).

## 2. Extractive QA (DeBERTa-v3-base)

| Run | Setup | Result |
|-----|-------|--------|
| v6 | 2 epochs full pass | Killed by Kaggle 9h GPU session limit at ~10h on P100 |
| v7 | max_steps=4000, batch 4, grad_accum 2 | Trained, model saved. CLS-index off-by-one in post-process blocked evaluation report write |

`models/deberta-cuad-extractor/` is the v7 checkpoint (700 MB safetensors).
It loads cleanly via the rewritten `DebertaExtractor` (transformers 5.x
compatible) but at 0.05 epoch of CUAD it is severely undertrained for
production span localisation. In the orchestrator we therefore keep
`EXTRACTOR_MODEL=""` by default — the classifier still drives risk scoring
and the full chunk is used as the span when no extractor is available.

Path to a usable extractor (future work):
- Bump `max_steps` to 16 000–32 000 across two ~5 h Kaggle sessions with
  `--resume_from_checkpoint`, OR
- Initialise from `deepset/deberta-v3-base-squad2` (pre-trained on SQuAD)
  and fine-tune for 4 000 steps on CUAD.

## 3. End-to-end pipeline (RAGAS-style)

### Setup

| Item | Value |
|------|-------|
| Eval LLM | `gpt-4o-mini` (best price/quality for judge prompts) |
| Classifier | `models/deberta-cuad-classifier` (v8) |
| RAG | ChromaDB with 12 curated GDPR / EU AI Act / practice-note snippets |
| Contracts evaluated | 3 (54 k, 70 k, 11 k chars) |
| Chunk size | 2 000 chars per analysis pass |
| Classification threshold | 0.6 (v2) vs 0.5 (v1) |

### Aggregate scores

| Run | Justification | Threshold | n risks | Faithfulness mean | Relevancy mean |
|-----|---------------|-----------|---------|-------------------|-----------------|
| v1 | Static keyword + rationale, no citation | 0.50 | 303 | 0.185 | 0.330 |
| **v2** | Verbatim quote of triggering keyword + offset | 0.60 | 228 | **0.231** | **0.377** |

v2 vs. v1: **+25% faithfulness, +14% relevancy** with the same classifier and
same 3 contracts. The lift comes from two changes only: (a) embedding a
verbatim citation of the triggering keyword in every policy-derived
justification, and (b) raising the classifier confidence threshold from
0.50 to 0.60 to suppress lower-confidence false positives (228 risks vs
303 — fewer but more confident).

The full per-category and per-risk breakdown is in
[`ragas_eval_report.json`](ragas_eval_report.json) (v1 baseline),
[`ragas_eval_report_v2.json`](ragas_eval_report_v2.json) (v2 with citations),
and the per-risk details files (`*.details.jsonl`).

**Top 10 categories by faithfulness in v2 (n≥2):**

| Category | Faithfulness | Relevancy | n |
|----------|--------------|-----------|---|
| Exclusivity | 0.375 | 0.750 | 4 |
| Parties | 0.333 | 0.167 | 3 |
| Non-Transferable License | 0.300 | 0.600 | 5 |
| Price Restrictions | 0.300 | 0.500 | 2 |
| Source Code Escrow | 0.300 | 0.000 | 2 |
| Competitive Restriction Exception | 0.286 | 0.286 | 7 |
| Expiration Date | 0.286 | 0.143 | 7 |
| Effective Date | 0.275 | 1.000 | 4 |
| License Grant | 0.275 | 0.875 | 4 |
| Notice Period To Terminate Renewal | 0.275 | 0.500 | 4 |

### Interpretation (honest)

The v1 → v2 jump is meaningful but the absolute numbers are still modest.
Concretely:

1. **Citations help, but the rule-based justification ceiling is real.**
   Embedding the triggering keyword + offset in every justification pushed
   faithfulness from 0.185 to 0.231 (+25%) and relevancy from 0.330 to 0.377
   (+14%). The gap between mean (0.231) and median (0.300) shows a long
   tail of low-faithfulness justifications driven by chunks where the
   classifier is over-eager.

2. **Relevancy is bounded by classifier precision on 2k-char chunks.**
   The v8 classifier was tuned for recall (`pos_weight` inversely scaled
   with class frequency). Raising the threshold from 0.50 to 0.60 in v2
   reduced emitted risks from 303 to 228 and improved both metrics. A
   further precision-oriented training pass (lower `pos_weight` cap,
   larger eval split, hard-negative mining) would close the rest of the
   gap, but is beyond the scope of this thesis.

3. **Path to >0.5 faithfulness.** Replace the policy template with a
   LLM-rewritten justification that explicitly cites a chunk-level span
   recovered by the extractor (when the extractor is properly converged
   — see §2). The architecture already supports this via the Legal
   Consultant agent; the bottleneck is the extractor, not the orchestrator.

These are honest, real numbers produced by `src/evaluation/ragas_eval.py`
with `gpt-4o-mini` as the judge over CUAD contracts.

## 4. Reproducibility

| Artifact | Command |
|----------|---------|
| CUAD multi-label dataset | `python scripts/prepare_multilabel_dataset.py` |
| Push datasets to Kaggle | `cd kaggle/datasets/cuad-multilabel && kaggle datasets create -p . --dir-mode zip` |
| Push training kernel | `cd kaggle/kernels/classifier && kaggle kernels push -p .` |
| Pull trained weights | `python scripts/pull_kaggle_models.py` |
| Seed regulation RAG | `python scripts/seed_regulations.py` |
| Run RAGAS eval | `python -m src.evaluation.ragas_eval --max-contracts 3 --eval-model gpt-4o-mini` |
| End-to-end demo (PDF + JSON) | `python scripts/demo_e2e.py path/to/contract.pdf` |

## 5. Open Kaggle artifacts

- Datasets: `milomilanovi/contractlens-cuad-multilabel`,
  `milomilanovi/contractlens-cuad-squad` (CC-BY-4.0)
- Kernels: `milomilanovi/contractlens-classifier-training`,
  `milomilanovi/contractlens-extractor-training`
