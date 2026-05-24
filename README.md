# ContractLens — Enterprise Legal AI Orchestrator

Hybrid system for automated legal risk analysis that combines a locally-trained
DeBERTa classifier with a multi-agent RAG pipeline. Every Risk Score is traced
back to a verbatim quote from the source contract, so the legal reviewer can
audit the model's reasoning end-to-end.

## Why this project

Standard "AI for legal" tools are black boxes. ContractLens replaces that with:

- **Privacy-first inference** — contracts never leave the box; only validated,
  redacted clauses can be sent to the cloud LLM, and the cloud step is fully
  optional (`DISABLE_LLM=1`).
- **Traceability** — every emitted `RiskScore` carries a `(span_start_offset,
  span_end_offset, source_doc)` triple. JSON and PDF compliance reports
  embed the same citations the policy engine used.
- **41 CUAD risk categories** — multi-label DeBERTa-v3-base fine-tuned on
  18 k windows + tuned per-class thresholds (Tuned micro F1 = 0.66, macro F1
  = 0.53 — see [docs/RESULTS.md](docs/RESULTS.md)).
- **Clean architecture** — Domain / Application / Infrastructure separation,
  Strategy pattern for LLM providers, Factory pattern for document parsers.

## Project layout

```
src/
├── domain/                          # Pure business logic (no infra deps)
│   ├── contract.py                  # Contract aggregate + ContractMetadata + Clause
│   ├── risk_policy.py               # 41-category default policy with citation-aware justifications
│   └── risk_score.py                # RiskScore value object (traceability fields)
├── application/                     # Use cases + ports (no vendor SDKs)
│   ├── generate_compliance_report.py
│   ├── evaluation/
│   │   └── evaluator.py             # LLMEvaluator (RAGAS faithfulness / relevancy)
│   ├── orchestration/
│   │   └── orchestrator.py          # LangGraph state machine — pipeline composition
│   └── interfaces/
│       ├── iclassifier.py
│       ├── iextractor.py
│       ├── illm_provider.py         # Strategy port for LLM vendors
│       ├── irisk_analyzer.py        # IRiskAnalyzer port (re-exports domain.RiskScore)
│       └── ivector_db.py
├── infrastructure/                  # Concrete adapters
│   ├── ai/
│   │   ├── hf_classifier.py         # HuggingFace + PEFT adapter loader
│   │   ├── deberta_extractor.py     # AutoModelForQuestionAnswering wrapper
│   │   ├── train_classifier.py
│   │   ├── train_extractor.py
│   │   └── kaggle_train_lora.py
│   ├── database/chroma_wrapper.py
│   ├── llm/openai_provider.py       # Strategy impl using openai>=1.0
│   └── reporting/pdf_renderer.py    # reportlab-based PDF compliance report
├── data/                            # Document normalization + sliding window
│   ├── document_normalizer.py       # Factory + TXT/MD/PDF/DOCX implementations
│   ├── sliding_window.py
│   └── cuad_loader.py
├── evaluation/
│   └── ragas_eval.py                # Batch RAGAS evaluation harness
└── api/
    └── main.py                      # FastAPI: /api/v1/analyze, /api/v1/report, /health

scripts/
├── prepare_multilabel_dataset.py    # CUAD SQuAD -> multi-label JSONL
├── seed_regulations.py              # Populate ChromaDB with regulatory snippets
├── pull_kaggle_models.py            # Download trained artifacts via kaggle CLI
└── demo_e2e.py                      # Parse PDF/DOCX/TXT -> JSON + PDF compliance report

kaggle/
├── kernels/{classifier,extractor}/  # Standalone Kaggle scripts (push via kaggle CLI)
└── datasets/{cuad-multilabel,cuad-squad}/dataset-metadata.json
```

## Quickstart

### 1. Local development

```bash
git clone https://github.com/milanovicmilos/contract-lens.git
cd contract-lens
python -m venv .venv
.venv/Scripts/activate     # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
cp .env.example .env       # add your OPENAI_API_KEY (cloud LLM is optional)
```

### 2. Run the CI checks before pushing

```bash
black src tests
ruff check src tests
pytest tests/ --cov=src
```

All three commands must pass — `.github/workflows/ci.yml` runs the same ones.

### 3. Analyse a contract end-to-end

```bash
# Populate the local RAG store (one-off — 48 article-level entries)
python scripts/seed_legal_corpus.py

# Parse a contract, classify, render compliance reports
python scripts/demo_e2e.py "CUAD_v1/full_contract_txt/<some_contract>.txt"
# -> reports/<stem>_compliance.json
# -> reports/<stem>_compliance.pdf
```

### 4. Start the API server

```bash
export CLASSIFIER_MODEL=models/deberta-cuad-classifier
export CHROMA_DIR=./chroma_db
# Generate a real API key once and set it in your env:
export CONTRACTLENS_API_KEYS=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
uvicorn src.api.main:app --reload
```

Endpoints:

| Method | Path | Body | Purpose |
|--------|------|------|---------|
| POST | `/api/v1/contracts` | multipart `file=<pdf\|docx\|txt\|md>` | Accept upload, run analysis in background, return `{job_id, status, status_url}` (202) |
| GET | `/api/v1/jobs/{job_id}` | — | Poll job status (`pending`/`running`/`completed`/`failed`); returns the RiskScore list on completion |
| POST | `/api/v1/analyze` | `{text, source_doc?}` | Synchronous; list of `RiskScoreResponse` (suited to short clauses only) |
| POST | `/api/v1/report` | `{text, source_doc?, format: "json"\|"pdf"}` | Compliance report path + summary |
| GET | `/health` | — | Readiness probe (no auth) |

All `/api/v1/*` routes require an `X-API-Key: <value>` header matching one
of `CONTRACTLENS_API_KEYS` (comma-separated). For local dev / CI you may
set `API_AUTH_DISABLED=1` — the server logs a loud WARNING when this is on.

The async upload flow is the right path for any contract longer than a
few clauses: the synchronous `/analyze` endpoint will time out on full
PDFs. `/api/v1/contracts` writes the job to a SQLite store
(`JOBS_DB_PATH`, default `data/jobs.sqlite`) and processes it in a
ThreadPoolExecutor (`JOB_WORKERS`, default 2). **Single-process only** —
scaling to multiple API workers needs Postgres + Celery/RQ.

The API starts even without `OPENAI_API_KEY` (the Legal Consultant just falls
back to rule-based notes); set `DISABLE_LLM=1` to suppress the cloud call
explicitly.

Other security knobs (see [`.env.example`](.env.example)): `RATE_LIMIT_ANALYZE`
(default `60/minute`), `RATE_LIMIT_REPORT` (default `10/minute`),
`RATE_LIMIT_UPLOAD` (default `10/minute`), `MAX_REQUEST_BODY_MB`
(default `5`), `ALLOWED_ORIGINS` (CORS allowlist; empty = no CORS).

## Architecture

The state machine in `src/application/orchestration/orchestrator.py` runs
four agents per chunk: Extractor → Validator → Legal Consultant (RAG + LLM)
→ Risk Auditor. Each step can degrade gracefully (no extractor, no LLM,
empty RAG corpus) without breaking the pipeline. The full diagram set with
the data-flow sequence diagram lives in [docs/arch.md](docs/arch.md).

## Models and metrics

See [docs/RESULTS.md](docs/RESULTS.md) for the complete F1 tables, RAGAS
faithfulness / relevancy aggregates, and an honest breakdown of which
categories work well and which do not.

| Component | Status |
|-----------|--------|
| v8 classifier (DeBERTa-v3-base + LoRA, 41 categories) | Tuned micro F1 = 0.662, macro F1 = 0.534 |
| v7 extractor (DeBERTa-v3-base QA) | Undertrained (5% of one epoch); kept disabled by default |
| Local RAG corpus | **48 article-level entries** (GDPR ×20, EU AI Act ×13, Practice Notes ×15) under `data/legal_corpus/*.jsonl`; seed via `scripts/seed_legal_corpus.py` |
| RAGAS faithfulness | See `docs/ragas_eval_report*.json` |

## LLM provider — OpenAI by default, swappable

The `ILLMProvider` Strategy interface (`src/application/interfaces/illm_provider.py`)
keeps the orchestrator and evaluator vendor-neutral. The current implementation
is `OpenAIProvider` (`src/infrastructure/llm/openai_provider.py`) using
`openai>=1.0` directly — no langchain dependency for production paths.

Default model is `gpt-4o-mini` (best price / quality balance for the legal
commentary and judge prompts we run). Override with `OPENAI_MODEL=gpt-4o`
when higher quality is worth the cost.

## Docker

```bash
docker compose up api
# or
docker build -f docker/Dockerfile --target inference -t contract-lens:latest .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e CLASSIFIER_MODEL=models/deberta-cuad-classifier \
  -v $(pwd)/models:/app/models \
  contract-lens:latest
```

## License

MIT — see [LICENSE](LICENSE).
