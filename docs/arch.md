# ContractLens — Architecture

This document describes the runtime architecture of ContractLens and how the
four pipeline agents collaborate through the LangGraph state machine.

## High-level layering

```mermaid
graph TB
    subgraph "API Layer (src/api/)"
        FA[FastAPI Server<br/>/api/v1/analyze, /health]
    end

    subgraph "Application Layer (src/application/)"
        UC[GenerateComplianceReport]
        OR[ContractOrchestrator<br/>LangGraph state machine]
        IF[Interfaces / Ports<br/>IClassifier, IExtractor,<br/>IVectorDatabase, ILLMProvider]
        EV[LLMEvaluator<br/>RAGAS faithfulness/relevancy]
    end

    subgraph "Domain Layer (src/domain/)"
        RP[RiskPolicy<br/>41 CUAD categories]
        RS[RiskScore<br/>+ traceability]
    end

    subgraph "Infrastructure Layer (src/infrastructure/)"
        HC[HFClassifier]
        DE[DebertaExtractor]
        CW[ChromaWrapper<br/>RAG]
        LLM[OpenAIProvider]
    end

    FA --> OR
    OR --> HC
    OR --> DE
    OR --> CW
    OR --> LLM
    OR --> RP
    OR --> RS
    UC --> IF
    EV --> LLM
    HC -.implements.-> IF
    DE -.implements.-> IF
    CW -.implements.-> IF

    style FA fill:#cfe2ff
    style RP fill:#ffe7c2
    style RS fill:#ffe7c2
    style OR fill:#d1e7dd
```

## Agent pipeline

The orchestrator implements a four-node state machine. The Validator can
short-circuit the pipeline before any LLM call is made, which keeps cost
bounded for noisy inputs.

```mermaid
graph LR
    START([Contract chunk]) --> EX[Extractor Agent<br/>classify + localize span]
    EX --> VA[Validator Agent<br/>min length + header regex]
    VA -- valid clause --> CO[Legal Consultant Agent<br/>RAG + LLM analysis]
    VA -- header/noise --> END_INVALID([end])
    CO --> AU[Risk Auditor Agent<br/>RiskPolicy + traceability]
    AU --> END_VALID([RiskScore list])

    style EX fill:#cfe2ff
    style VA fill:#fff3cd
    style CO fill:#d1e7dd
    style AU fill:#f8d7da
```

## Why each node exists

| Node | Spec mapping | Failure mode | Degraded behavior |
|------|--------------|--------------|-------------------|
| Extractor | Local DeBERTa for span extraction (privacy-first) | Extractor not loaded | Uses full chunk as span; offsets become None |
| Validator | Header/noise filter | Always succeeds; conservative thresholds | n/a |
| Consultant | OpenAI + RAG for reasoning | No API key / network failure | Rule-based justification only |
| Auditor | RiskPolicy + 41-category rules + traceability | Always succeeds | n/a |

## Privacy-first data flow

```mermaid
sequenceDiagram
    participant U as User / API caller
    participant API as FastAPI
    participant OR as Orchestrator
    participant LOC as Local DeBERTa (Classifier+Extractor)
    participant CH as ChromaDB (local RAG)
    participant CL as OpenAI (cloud)

    U->>API: POST /api/v1/analyze {text, source_doc}
    API->>OR: orchestrator.analyze(text)
    OR->>LOC: classify(text) + extract spans
    LOC-->>OR: classifications, span offsets
    Note over OR: Validator filters out headers
    OR->>CH: search(text, top_k=3) [LOCAL]
    CH-->>OR: regulation snippets
    OR->>CL: invoke(clause + RAG snippets)
    Note over CL: Only the validated clause<br/>+ regulation context leaves device
    CL-->>OR: consultant analysis
    OR->>OR: RiskPolicy.assess_risk per category
    OR-->>API: RiskScore[] (with span offsets, source_doc)
    API-->>U: 200 JSON
```

Key invariants:
- Raw contract bytes never leave the local machine.
- Only validated clauses (those passing the Validator) reach the cloud LLM.
- RAG corpus is embedded and searched entirely on-device.
- The full chunk content is sent to the LLM only when explicitly required
  for Consultant reasoning; if `DISABLE_LLM=1` is set, no cloud call is made.

## Configuration (env vars)

Set in `.env` or process environment. All have safe defaults.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLASSIFIER_MODEL` | `prajjwal1/bert-tiny` | HF identifier or local path for the classifier |
| `EXTRACTOR_MODEL` | `` (disabled) | Path or HF id for DeBERTa extractor; empty disables |
| `CHROMA_DIR` | `./chroma_db` | On-disk Chroma persistence path |
| `CHROMA_COLLECTION` | `legal_regulations` | Collection name |
| `OPENAI_API_KEY` | unset | If set, enables the Legal Consultant LLM call |
| `OPENAI_MODEL` | `gpt-4o` | Model used by Consultant |
| `DISABLE_LLM` | `false` | Force-disable LLM even if API key is present |
| `TESTING` | `false` | Skip orchestrator init at startup (used by tests) |

## Reproducible training (Kaggle)

```mermaid
graph LR
    L[scripts/prepare_multilabel_dataset.py] --> M[data/processed/cuad_multilabel.jsonl]
    M -- kaggle CLI --> KM[Kaggle Dataset<br/>contractlens-cuad-multilabel]
    SQ[data/processed/cuad_squad.jsonl] -- kaggle CLI --> KSQ[Kaggle Dataset<br/>contractlens-cuad-squad]
    KM --> NB1[kaggle/notebook_classifier.py]
    KSQ --> NB2[kaggle/notebook_extractor.py]
    NB1 -- T4 x2, 4 epochs --> W1[deberta-cuad-classifier/]
    NB2 -- T4 x2, 2 epochs --> W2[deberta-cuad-extractor/]
    W1 --> API
    W2 --> API
```

To trigger production training:
1. `kaggle datasets create -p kaggle/datasets/cuad-multilabel`
2. Open a Kaggle Notebook with both datasets attached and GPU T4 x2
3. Paste `kaggle/notebook_classifier.py` (or `notebook_extractor.py`) as a cell
4. Run. The final cell saves to `/kaggle/working/<model_dir>` and writes
   `eval_report.json` with per-category F1.

## Evaluation flow

```mermaid
graph TB
    DS[CUAD test set<br/>JSONL] --> RE[src/evaluation/ragas_eval.py]
    RE --> OR2[Orchestrator]
    OR2 --> RS2[RiskScore stream]
    RS2 --> EV2[LLMEvaluator]
    EV2 -- faithfulness --> RPT[evaluation_report.json]
    EV2 -- relevancy --> RPT
    RS2 -- raw --> DET[evaluation_details.jsonl]

    style RE fill:#cfe2ff
    style RPT fill:#d1e7dd
```

## File map

```
src/
├── api/
│   └── main.py                          # FastAPI + lifespan + orchestrator
├── application/
│   ├── generate_compliance_report.py    # JSON/PDF reporter use case
│   ├── evaluation/
│   │   └── evaluator.py                 # LLMEvaluator (RAGAS faithfulness)
│   ├── orchestration/
│   │   └── orchestrator.py              # LangGraph state machine (pipeline composition)
│   └── interfaces/                      # IClassifier, IExtractor, ILLMProvider, IVectorDatabase, IRiskAnalyzer
├── domain/
│   ├── contract.py                      # Contract aggregate + Clause + ContractMetadata
│   ├── risk_policy.py                   # 41 CUAD category policies
│   └── risk_score.py                    # RiskScore value object (traceability fields)
├── evaluation/
│   └── ragas_eval.py                    # batch eval harness
├── infrastructure/
│   ├── ai/
│   │   ├── hf_classifier.py             # HuggingFace classifier wrapper
│   │   ├── deberta_extractor.py         # DeBERTa QA extractor
│   │   ├── train_classifier.py          # Local fine-tuning
│   │   ├── train_extractor.py           # Local fine-tuning (QA)
│   │   └── kaggle_train_lora.py         # Kaggle entry point
│   ├── database/
│   │   └── chroma_wrapper.py            # ChromaDB RAG store
│   ├── llm/
│   │   └── openai_provider.py           # OpenAI Strategy implementation
│   └── reporting/
│       └── pdf_renderer.py              # reportlab-based PDF compliance report
└── data/
    ├── cuad_loader.py                   # CUAD JSON + CSV ingest
    ├── document_normalizer.py           # PDF/DOCX -> markdown
    └── sliding_window.py                # overlapping chunk tokenization

scripts/
├── analyze_cuad_structure.py
├── prepare_squad_dataset.py
├── prepare_multilabel_dataset.py        # SQuAD -> multi-label converter
├── push_to_kaggle.py
└── seed_regulations.py                  # populate ChromaDB with GDPR/AI Act

kaggle/
├── notebook_classifier.py               # standalone Kaggle script
├── notebook_extractor.py                # standalone Kaggle script
└── datasets/
    ├── cuad-multilabel/dataset-metadata.json
    └── cuad-squad/dataset-metadata.json
```
