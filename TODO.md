**ContractLens — Production Readiness Tracker**

Plan rada je definisan u [spec.txt](spec.txt#L1-L126) na osnovu CUAD dataset-a
([CUAD_v1/CUAD_v1_README.txt](CUAD_v1/CUAD_v1_README.txt#L1-L200)).

Status oznake: ✅ završeno, 🟡 u toku, ⏳ čeka spoljni resurs, ⬜ nije počelo.

---

## FAZA 1 — Data Pipeline ✅

- ✅ `src/data/document_normalizer.py` — PDF/TXT → markdown čišćenje
- ✅ `src/data/sliding_window.py` — overlapping chunk tokenization
- ✅ `src/data/cuad_loader.py` — CUAD JSON + master_clauses.csv merge sa fuzzy filename match
- ✅ `scripts/prepare_squad_dataset.py` → `data/processed/cuad_squad.jsonl` (20,910 SQuAD rows)
- ✅ `scripts/prepare_multilabel_dataset.py` → `data/processed/cuad_multilabel.jsonl` (18,176 examples, 13,823 pozitivnih + 4,353 negativnih, sve 41 kategorije)

## FAZA 2 — Local Extraction PoC ✅

- ✅ Interfejs `IExtractor` + `ExtractionResult` dataclass
- ✅ `src/infrastructure/ai/deberta_extractor.py` — DeBERTa QA implementacija
- ✅ `src/infrastructure/ai/train_extractor.py` — fine-tuning skripta
- ✅ Standalone Kaggle skripta `kaggle/notebook_extractor.py` (sm_60+sm_75 GPU kompatibilna)

## FAZA 3 — Multi-label Classification ✅

- ✅ Interfejs `IClassifier`
- ✅ `src/infrastructure/ai/hf_classifier.py` — HF pipeline + LoRA adapter loading + 41-CUAD label remap
- ✅ `src/infrastructure/ai/train_classifier.py` — pravi Trainer loop sa sklearn micro/macro F1, EarlyStopping, per-category classification_report
- ✅ `src/infrastructure/ai/kaggle_train_lora.py` — produkcijska Kaggle skripta sa LoRA r=16, alpha=32
- ✅ `src/domain/risk_policy.py` — default policy za svih 41 kategorija (rationale + high-risk keywords)

## FAZA 4 — Agent Orchestration ✅

- ✅ `src/application/orchestration/orchestrator.py` — LangGraph state machine (Extractor → Validator → Consultant → Auditor)
  - Validator radi i ML word-count i regex header detekciju
  - Extractor se poziva per-kategorija za span localization
  - Consultant ima graceful fallback kad nema LLM klijenta
  - Auditor proizvodi RiskScore sa span offsets i source_doc
- ✅ `src/infrastructure/database/chroma_wrapper.py` — RAG store sa graceful chromadb fallback
- ✅ `scripts/seed_regulations.py` — 12 GDPR/EU AI Act snippeta indeksirano u ChromaDB

## FAZA 5 — Kaggle Trening (LoRA) ⏳

- ✅ Kaggle datasets uploadovani: `milomilanovi/contractlens-cuad-multilabel`, `milomilanovi/contractlens-cuad-squad`
- ✅ Kaggle kernels pushed: `contractlens-classifier-training`, `contractlens-extractor-training`
- ✅ Kernel metadata + scripts kompatibilne sa sm_60 (P100) + sm_75 (T4) GPU
- ✅ `scripts/pull_kaggle_models.py` — preuzima trainovane modele + piše `models/MANIFEST.json`
- ⏳ Trenutni Kaggle run (v5) — transformers 4.46 + torch 2.4 + DeBERTa-v3-base sa safetensors
- ⬜ Integracija težina u API (postavljanje `CLASSIFIER_MODEL=models/deberta-cuad-classifier`)

## FAZA 6 — Evaluation & Traceability ✅

- ✅ `RiskScore` ima `span_start_offset`, `span_end_offset`, `source_doc` polja
- ✅ `src/application/evaluation/evaluator.py` — RAGAS faithfulness sa LLM-as-judge JSON parsing
- ✅ `src/evaluation/ragas_eval.py` — batch evaluacija sa per-category + aggregate metrics
- ✅ Smoke test pipeline: 35 RiskScore-ova emitovanih iz jednog ugovora
- ⬜ Final eval report sa fine-tuned modelom + OpenAI judge

## FAZA 7 — API, Docker, CI/CD ✅

- ✅ `src/api/main.py` — FastAPI + lifespan, opcionalan OPENAI_API_KEY, env-driven config
- ✅ Multi-stage Dockerfile (dev / inference / training)
- ✅ GitHub Actions CI: black, ruff, pytest, bandit
- ✅ 43 unit + integration testa prolazi

## FAZA 8 — Documentation & Thesis ✅ docs, ⬜ thesis

- ✅ `docs/arch.md` — Mermaid dijagrami arhitekture, agent flow, privacy data-flow, Kaggle pipeline
- ⬜ Final thesis chapters

---

## DELIVERABLES STATUS

- ✅ Slojevit `src/` (domain, application, infrastructure, api, evaluation, data)
- ✅ Data pipeline `data/processed/cuad_squad.jsonl` (1.1GB) + `cuad_multilabel.jsonl` (38MB)
- ⏳ Trained model weights (čeka Kaggle)
- ⬜ `docs/evaluation_report.json` sa F1 ≥ 0.85 (čeka modele)
- ✅ Dockerfile, GitHub Actions, kompletan README + arch docs

## ACCEPTANCE CRITERIA

- ✅ Reproducible pipeline od raw → processed → eval skripte spremne
- ⏳ Extractive QA F1 > 0.85 (cilj) — čeka Kaggle trening
- ⏳ Multi-label F1 > 0.85 (cilj) — čeka Kaggle trening
- ✅ Svaki RiskScore vraća span + offset + source_doc (traceability potvrđena u testovima)

---

## TRENUTNI STATUS (2026-05-22)

**Šta radi sad:** Sve od arhitekture, API-a, RAG-a, evaluacijskog pipeline-a, do CI/CD-a je gotovo
i testirano (43/43 testa). Domain layer pokriva svih 41 CUAD kategorija. Orchestrator radi
end-to-end u offline modu (35 risk score-ova iz jednog test ugovora).

**Šta čeka:** Kaggle treninzi v5 (transformers 4.46 + torch 2.4 + DeBERTa-v3-base sa
safetensors) — kad završe, `scripts/pull_kaggle_models.py` povlači težine i piše
`models/MANIFEST.json`, a zatim `python -m src.evaluation.ragas_eval` (sa
`OPENAI_API_KEY`) generiše finalni F1 + faithfulness/relevancy report za tezi.

**Branch:** `feat/production-readiness` na origin-u, otvoren za PR.
