# ContractLens — Training & Evaluation Results

This document is the **single source of truth** for every metric quoted in
the master's thesis. All numbers are produced by reproducible scripts in
this repository; no figure here is invented.

## 1. Multi-label Classifier (DeBERTa-v3-base + LoRA)

### Setup (v9 — currently shipping)

| Item | Value |
|------|-------|
| Base model | `microsoft/deberta-v3-base` (138 M params) |
| Adapter | LoRA on `query_proj` / `key_proj` / `value_proj`, **r=64, α=128** |
| Trainable parameters | 3.6 M (1.95% of total — wider adapter than v8) |
| Loss | BCE with `pos_weight` per class (inverse frequency, **clipped to 50**) |
| Epochs | **12** with EarlyStopping (patience 4) on micro F1 |
| Train / eval split | 18 176 windows → 16 358 train / 1 818 eval |
| Threshold | per-class F1-optimal on the eval set |
| Hardware | Kaggle GPU (T4 or P100 — both supported via torch 2.4 + transformers 4.46) |

### Iteration history

| Run | Δ | Tuned micro F1 | Tuned macro F1 |
|-----|---|----------------|----------------|
| v6 | baseline (no class weights, 4 epochs) | 0.461 | 0.083 |
| v7 | + class-weighted BCE, r=32, 8 epochs, per-class threshold tuning | 0.671 | 0.561 |
| v8 | + merge LoRA into base for portable inference | 0.662 | 0.534 |
| v9.0 | attempted DeBERTa-v3-large + r=64 + grad-accum=8 | **silent OOM** (zero-byte log; see ADR-002) |
| v9.1 | reverted to -base, kept r=64 + α=128 + pos_weight cap=50 + 12 epochs | (POS_WEIGHT_CAP NameError pre-import; see PR #8) |
| **v9.2** | + hoisted POS_WEIGHT_CAP fix; full 12-epoch run (4.3 h on T4) | **0.688** | **0.579** |

v9.2 is what ships in `models/deberta-cuad-classifier/` (the merged 737 MB
artifact). The lift from v8 → v9.2 is **+2.6 pp micro, +4.5 pp macro** — modest
but real, and concentrated where it matters most (rare categories, see below).
v9.0 and v9.1 are documented failures kept in the history so the reader can
trace what works and what does not; the v9.0 silent-OOM incident is the
substance of [ADR-002](adr/002-lora-on-deberta-base-not-full-fine-tune.md).

### Per-category F1 (v9.2, tuned thresholds)

**Top 15 (F1 ≥ 0.68):**

| Category | F1 | Precision | Recall | Support |
|----------|----|-----------|--------|---------|
| Parties | 0.929 | 0.930 | 0.928 | 401 |
| Third Party Beneficiary | 0.909 | 0.938 | 0.882 | 17 |
| Document Name | 0.902 | 0.865 | 0.943 | 333 |
| Agreement Date | 0.885 | 0.836 | 0.940 | 331 |
| Expiration Date | 0.845 | 0.786 | 0.914 | 209 |
| Insurance | 0.836 | 0.875 | 0.800 | 70 |
| License Grant | 0.828 | 0.781 | 0.882 | 238 |
| Renewal Term | 0.782 | 0.684 | 0.912 | 102 |
| Cap On Liability | 0.781 | 0.812 | 0.752 | 121 |
| Governing Law | 0.771 | 0.763 | 0.779 | 95 |
| Irrevocable Or Perpetual License | 0.748 | 0.678 | 0.836 | 73 |
| Effective Date | 0.736 | 0.669 | 0.818 | 274 |
| Warranty Duration | 0.725 | 0.659 | 0.806 | 36 |
| Notice Period To Terminate Renewal | 0.702 | 0.570 | 0.914 | 58 |
| Non-Compete | 0.685 | 0.612 | 0.778 | 81 |

**Categories where v9.2 lifted v8 most:**

| Category | v8 F1 | v9.2 F1 | Δ | Support |
|----------|-------|---------|---|---------|
| Third Party Beneficiary | ~0.43 | **0.909** | +47 pp | 17 |
| Insurance | 0.668 | **0.836** | +17 pp | 70 |
| Cap On Liability | 0.669 | **0.781** | +11 pp | 121 |
| Governing Law | ~0.65 | **0.771** | +12 pp | 95 |

The tighter `pos_weight=50` (was 100) reduced noise saturation on
mid-frequency categories, which is exactly where the lift landed.

**Weakest (still under F1 0.30 — tiny eval supports):**

| Category | v8 F1 | v9.2 F1 | Support | Honest read |
|----------|-------|---------|---------|-------------|
| Most Favored Nation | 0.146 | 0.057 | 7 | **Regressed.** 7 positives is below the noise floor — any single misclassification swings F1 ±0.15. Statistically meaningless either direction. |
| Price Restrictions | 0.111 | 0.222 | 4 | n=4. Doubling is one extra correct prediction. |
| Source Code Escrow | ~0.25 | 0.231 | 10 | Sparse training signal regardless of run. |
| Non-Disparagement | ~0.20 | 0.259 | 16 | Same. |
| Change Of Control | 0.275 | 0.276 | 56 | High lexical variance — only a different model class would move this. |

Of 41 categories, **22 reach F1 ≥ 0.60** at tuned thresholds in v9.2
(up from 17 in v8) — the operationally useful subset for the API demo
and risk dashboard.

Full report: [`classifier_eval_report.json`](classifier_eval_report.json)
(v8 baseline) and the v9.2 report under
`models/eval_report_v9_2.json`.

## 2. Extractive QA

| Run | Setup | Token F1 | Exact match | n eval |
|-----|-------|----------|-------------|--------|
| v6 | DeBERTa-v3-base, 2 epochs full pass | — | — | Killed by Kaggle 9h limit |
| v7 | DeBERTa-v3-base from scratch, max_steps=4000 | — | — | CLS-index bug blocked eval write |
| **v8** | **`deepset/deberta-v3-base-squad2` init + max_steps=8000 + CLS-index fix** | **0.277** | **0.231** | **2 091** |

v8 trained cleanly in 3.5 h on a Kaggle T4, with the SQuAD2-pretrained QA
head adapting to CUAD over 8 000 steps (≈ 0.11 of one full pass through
the sliding-window dataset). Final train loss 0.163 — the head learned to
fit CUAD spans, but token-level F1 = 0.28 is well below production grade.

**Honest interpretation:** CUAD answer spans are dramatically longer than
SQuAD2's typical 1–5 token answers (CUAD averages 30–80 tokens). The
pretrained head's distribution of start/end logits is calibrated for the
shorter span regime, and 8 000 steps is not enough to retrain that
distribution end-to-end. The extractor is therefore **disabled by default**
in the orchestrator (`EXTRACTOR_MODEL=""`); the classifier drives risk
scoring and the validator chunk is used as the span when no extractor is
available.

The v8 model artifact (737 MB safetensors) plus optimizer state at
checkpoint-8000 is preserved for a future v9 extractor: resume training
for another 16–24 k steps with a span-length-aware loss (or with
[layout-aware CUAD-pretrained checkpoints](https://huggingface.co/atticus)
when those land) should close the gap. See PR #4 commit notes for the
v8 root-cause analysis.

## 3. End-to-end pipeline (RAGAS-style)

### Setup (v3 — currently shipping)

| Item | Value |
|------|-------|
| Eval LLM | `gpt-4o-mini` (best price/quality for judge prompts) |
| Classifier | `models/deberta-cuad-classifier` (**v9.2**) |
| RAG corpus | **48 article-level entries** — GDPR ×20, EU AI Act ×13, Practice Notes ×15 (`data/legal_corpus/`) |
| Justification | **LLM-rewritten, RAG-grounded** via `src/application/llm_justifier.py` (one JSON-mode call per chunk) |
| Contracts evaluated | 3 (54 k, 70 k, 11 k chars) |
| Chunk size | 2 000 chars per analysis pass |
| Classification threshold | 0.6 |

### Aggregate scores — iteration history

| Run | Classifier | RAG corpus | Justification | n risks | Faithfulness mean | Faithfulness median | Relevancy mean |
|-----|-----------|------------|---------------|---------|------|------|------|
| v1 | v8 | 12 inline snippets | Static template, no citation | 303 | 0.185 | 0.200 | 0.330 |
| v2 | v8 | 12 inline snippets | Verbatim quote of triggering keyword + offset | 228 | 0.231 | 0.300 | 0.377 |
| **v3** | **v9.2** | **48 article-level entries** | **LLM-rewritten, RAG-grounded** | **172** | **0.513** | **0.700** | **0.401** |

**v3 vs. v2 (the headline):**

- Faithfulness mean: **0.231 → 0.513 (+122%)** — crosses the "Path to >0.5"
  threshold the v2 writeup flagged.
- Faithfulness median: **0.300 → 0.700 (+133%)** — the typical justification
  is now strongly faithful, not borderline.
- Relevancy mean: 0.377 → 0.401 (+6%, modest lift).
- Risks emitted: 228 → 172 — fewer because v9.2's tighter pos_weight
  produces more precise classifications at the same 0.6 threshold.

**Top 10 categories by faithfulness in v3 (n ≥ 3):**

| Category | Faithfulness | Relevancy | n |
|----------|--------------|-----------|---|
| License Grant | 1.000 | 0.833 | 3 |
| Anti-Assignment | 0.800 | 0.700 | 5 |
| Expiration Date | 0.800 | 0.167 | 6 |
| Change Of Control | 0.783 | 0.500 | 6 |
| Cap On Liability | 0.764 | 0.500 | 11 |
| Competitive Restriction Exception | 0.750 | 0.375 | 4 |
| IP Ownership Assignment | 0.725 | 0.500 | 4 |
| Exclusivity | 0.720 | 0.960 | 5 |
| Non-Transferable License | 0.667 | 0.500 | 3 |
| Non-Compete | 0.567 | 0.500 | 3 |

Cap On Liability — n=11 (statistically meaningful) at faithfulness 0.764
is the strongest single signal that the LLM rewriter + RAG corpus + better
classifier compound on top of each other.

### Interpretation (honest)

The v2 → v3 lift was not free — three changes compounded:

1. **v9.2 classifier raised the precision floor** of what the orchestrator
   even classifies as a risk (micro F1 0.662 → 0.688, macro 0.534 → 0.579;
   the rare categories that v8 hallucinated are gone).
2. **48-article RAG corpus** lets the LLM cite a specific regulation
   ("Per GDPR Art. 28, the clause …") instead of a 12-snippet paraphrased
   blob. Citation discipline is enforced by the system prompt and is what
   makes the median faithfulness so much higher than the mean.
3. **One LLM call per chunk** (not per risk) keeps API spend flat per
   contract while the JSON-mode response gives one grounded justification
   per detected category.

**Remaining gap.** Mean (0.513) below median (0.700) shows a long tail of
chunks where the LLM was unable to produce a faithful justification —
typically because the classifier flagged the wrong category and the LLM
honestly couldn't ground it. A precision-oriented v10 classifier pass
(hard-negative mining + larger eval split) would close this. Out of scope
for this thesis.

**Relevancy didn't lift as much as faithfulness.** Relevancy measures
whether the extracted span demonstrates the category concept; with the
extractor still disabled by default (see §2), the "extracted span" is the
whole 2 k chunk and includes a lot of off-topic context. Lifting the
extractor to a usable F1 would directly improve this metric.

Full v3 report:
- [`docs/ragas_eval_report_v3.json`](ragas_eval_report_v3.json) — aggregate + per-category.
- [`docs/ragas_eval_report_v3.details.jsonl`](ragas_eval_report_v3.details.jsonl) — one row per risk with faithfulness + relevancy scores.
- Historical: [`ragas_eval_report.json`](ragas_eval_report.json) (v1), [`ragas_eval_report_v2.json`](ragas_eval_report_v2.json) (v2).

All numbers produced by `src/evaluation/ragas_eval.py` with `gpt-4o-mini`
as the judge over real CUAD contracts. Reproducible via `inv eval --max-contracts 3`.

## 4. Reproducibility

All commands run via the `inv` task runner (`tasks.py`). Equivalent direct
invocations shown alongside.

| Artifact | `inv` command | Direct |
|----------|---------------|--------|
| Install dependencies | `inv install` | `pip install -r requirements-dev.txt` |
| CUAD multi-label dataset | `inv data` | `python scripts/prepare_multilabel_dataset.py` |
| Push datasets to Kaggle | — | `cd kaggle/datasets/cuad-multilabel && kaggle datasets create -p . --dir-mode zip` |
| Push training kernel | — | `cd kaggle/kernels/classifier && kaggle kernels push -p .` |
| Pull trained weights | `inv pull-models` | `python scripts/pull_kaggle_models.py` |
| Seed regulation RAG (48 articles) | `inv seed` | `python scripts/seed_legal_corpus.py` |
| Run RAGAS eval (v3 setup) | `inv eval --max-contracts 3` | `python -m src.evaluation.ragas_eval --max-contracts 3 --eval-model gpt-4o-mini --chroma-dir ./chroma_db` |
| End-to-end demo (PDF → JSON + PDF) | `inv demo --contract <path>` | `python scripts/demo_e2e.py <path>` |
| Start API server | `inv api` | `uvicorn src.api.main:app` |
| Full CI quality gates | `inv ci` | `black --check + ruff check + pytest + bandit` |

## 5. Open Kaggle artifacts

- Datasets: `milomilanovi/contractlens-cuad-multilabel`,
  `milomilanovi/contractlens-cuad-squad` (CC-BY-4.0)
- Kernels: `milomilanovi/contractlens-classifier-training`,
  `milomilanovi/contractlens-extractor-training`
