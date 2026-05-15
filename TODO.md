**ContractLens — Detaljan TODO plan**

Ovaj fajl sadrži strukturisan, fazni i veoma detaljan plan rada za master rad zasnovan na specifikaciji [spec.txt](spec.txt#L1-L126) i CUAD dataset-u ([CUAD_v1/CUAD_v1_README.txt](CUAD_v1/CUAD_v1_README.txt#L1-L200), [CUAD_v1/master_clauses.csv](CUAD_v1/master_clauses.csv#L1-L10)).

FAZE I ZADACI

1) Analiza i priprema podataka (1–2 nedelje)
- **Cilj:** Razumeti CUAD strukturu, očistiti i normalizovati tekstove.
- **Koraci:**
  - **Pročitati i verifikovati:** otvoriti [CUAD_v1/CUAD_v1_README.txt](CUAD_v1/CUAD_v1_README.txt#L1-L200) i `master_clauses.csv` (primer: [CUAD_v1/master_clauses.csv](CUAD_v1/master_clauses.csv#L1-L10)).
  - **DocumentNormalizer:** napisati skriptu `src/data/document_normalizer.py` koja:
    - parsira TXT/PDF -> canonical plain text/markdown,
    - uklanja footere, page headers, višestruke razmake, i normalizuje redne datume,
    - označava stranice i paragraf-id (potrebno za extractive QA span koordinatu).
  - **CSV -> SQuAD pipeline:** iskoristiti postojeći SQuAD-like JSON u `CUAD_v1.json` ili generisati iz `master_clauses.csv` skriptom `src/data/csv_to_squad.py`.
  - **SlidingWindowTokenization:** implementirati `src/data/sliding_window.py` koji deli tekst u preklapajuće blokove sa metas (doc, page, offset).
  - **Output:** `data/processed/cuad_squad.jsonl` i `data/processed/paragraph_index.parquet`.

2) Brzi prototip lokalne ekstrakcije (2–3 nedelje)
- **Cilj:** Lokalno izvući kandidovane pasuse i osnovne spans (extractor).
- **Koraci:**
  - **Model izbor:** koristite manji, brzi encoder (npr. DeBERTa-base / LegalBERT) za PoC; kasnije zameniti DeBERTa-v3-Legal-Large za finalnu izvedbu.
  - **Trening / Fine-tune:** `src/ai/train_extractor.py` — fine-tune za SQuAD-like extractive QA nad `cuad_squad.jsonl` (koristiti Hugging Face / transformers).
  - **Inference skripta:** `src/ai/extract.py` vraća top-K kandidatnih spanova sa skorovima i kontekstnim metama.
  - **Lokacija rada:** sve lokalno (privatnost + brz iterativni razvoj).

3) Multi-label klasifikacija & risk-scoring local pipeline (2–3 nedelje)
- **Cilj:** Detektovati presence/absence (Yes/No) i višestruke kategorije za paragraf.
- **Koraci:**
  - **Model:** multi-label klasifikator (binary head per label) `src/ai/train_classifier.py`.
  - **RiskPolicy engine:** `src/domain/risk_policy.py` — implementirati pravila (weights, thresholds) i interfejs `IRiskAnalyzer`.
  - **Unit tests:** `tests/test_risk_scoring.py` — pokriti logiku thresholda i pravila.

4) Agentna orkestracija i RAG (Reasoning + Evidence) (3–4 nedelje)
- **Cilj:** Implementirati agent flow: Extractor → Validator → LegalConsultant → RiskAuditor.
- **Koraci:**
  - **LangGraph / Orchestrator skeleton:** `src/infrastructure/agents/orchestrator.py` (state-machine), lokalno testirati sa stub LLM.
  - **Validator Agent:** pravila + ml heuristike za razlikovanje naslov vs clause.
  - **RAG:** indeksi u lokalnom ChromaDB (`src/infrastructure/database/chroma_wrapper.py`) za pravne reference (regulative).
  - **Legal Consultant agent:** Implementacija integracije sa **OpenAI API** mrežom (npr. gpt-4o/gpt-4-turbo) za pravna poređenja i reasoning. Definisati RAG pipeline tako da se ka OpenAI-u šalju samo relevantni ili anonimizovani delovi zbog privatnosti ugovora.

5) Kaggle faza — veliki trening i LoRA/QLoRA (eksperimentalno) (2–4 nedelje na Kaggle)
- **Cilj:** Iskoristiti dostupne T4 GPU na Kaggle za efikasan fine-tuning (LoRA/QLoRA) modela DeBERTa-v3-Legal-Large ili sličnih.
- **Koraci:**
  - **Priprema:** paketovati `data/processed/` i training skripte u reproducible notebook / script `kaggle/train_lora.py`.
  - **Tehnika:** LoRA ili QLoRA (ako mem. ograničenje) — koristiti bitsandbytes + peft.
  - **Metrika:** pratiti F1, EM za extractive QA i micro/macro F1 za multi-label; cilj F1>0.85 na CUAD test setu.
  - **Artifacts:** model weights (LoRA adapteri), eval report, training logs; prebaciti najbolje rezultate u `models/` (ne uključivati sirove težine velike > git ignore).

6) Evaluation, interpretability i traceability (2 nedelje)
- **Cilj:** Svaki Risk Score da bude potkrepljen citatom i zakon reference.
- **Koraci:**
  - **Traceable Evidence:** za svako predviđanje vraćati span + skor + source doc + offset.
  - **LLM Evaluation (RAGAS):** implementirati skripte za faithfulness i relevancy evaluaciju `src/evaluation/ragas_eval.py`.
  - **Human-in-the-loop:** CSV/Excel export za pravnu reviziju klauzula.

7) API, CI/CD, Docker i dokumentacija (1–2 nedelje)
- **Cilj:** Dostaviti reproducibilan razvojni i produkcioni stack.
- **Koraci:**
  - **FastAPI endpoint:** `src/web_api/main.py` sa endpointima: `/analyze`, `/explain`, `/health`.
  - **Dockerfile:** multi-stage Dockerfile za aplikaciju i manju image za inference; dodati `docker/`.
  - **GitHub Actions:** CI za lint (ruff/black), unit testove i osnovni integration smoke test.
  - **README & docs:** `docs/arch.md`, `docs/run_local.md`, i mermaid dijagrami.

8) Finalni rad i prezentacija (2 nedelje)
- **Cilj:** Složiti master rad, rezultate, evaluacije i demo.
- **Koraci:**
  - **Poglavlja rada:** Uvod, Dataset, Metodologija, Modeli, Evaluacija, Diskusija, Zaključak, Budući rad.
  - **Prilozi:** kodni repo snapshot, modeli (link na weight storage), eval izvještaji, demo video/skripte.

PRIORITETI I RASPODJELA: LOKALNO VS KAGGLE VS CLOUD
- **Obavezno lokalno (Privacy-First):**
  - Document normalizacija, tokenizacija, trening-inferenca PoC, RAG baza (lokalna Chroma), API development i unit testovi.
- **Kaggle (GPU-heavy, reproducible):**
  - Veliki fine-tuning (LoRA/QLoRA) na T4 x2 (Kaggle). Tu izvodite trening ekstrakcije i klasifikacije kako biste sačuvali lokalno vreme i resurse. Težine (adapters) se skidaju za lokalnu inferencu.
- **Cloud / API (OpenAI za reasoning):**
  - **OpenAI API** – korišćenje najjačih komercijalnih modela kao agent-mozga za Legal Consultant deo. Model procesaira klasifikovane klauzule za naprednu RAG interpretaciju i generisanje reasoning-a.

DELIVERABLES (minimalno za odbranu master rada)
- Kod u `src/` sa jasnom podelom slojeva (domain, application, infrastructure, web_api).
- Data pipeline u `src/data/` i `data/processed/` prikaz primerka.
- Trained LoRA adapteri i eval report (artifacts). Ne dodavati velike binarne fajlove u git.
- Dockerfile, GitHub Actions workflow, i `README.md` sa uputstvima.

ACCEPTANCE CRITERIA
- Reproducibilni pipeline: od raw TXT -> processed -> model eval.
- Extractive QA: F1 > 0.85 (cilj), minimalno F1 > 0.75 za Proof-of-Concept.
- Svaki Risk Score vraća span + citat + izvor (doc name + offset).

NAPOMENA O PRIVATNOSTI I LICENCI
- CUAD je pod CC BY 4.0 — koristite u skladu sa licencom.
- Osetljive ugovorne tekstove držati lokalno; slati u cloud samo anonimizovane ili agregirane rezultate.

Kratka lista fajlova koje odmah kreirati/implementirati:
- `src/data/document_normalizer.py`
- `src/data/sliding_window.py`
- `src/data/csv_to_squad.py`
- `src/ai/train_extractor.py`
- `src/ai/extract.py`
- `src/ai/train_classifier.py`
- `src/domain/risk_policy.py`
- `src/infrastructure/agents/orchestrator.py`
- `src/infrastructure/database/chroma_wrapper.py`
- `src/web_api/main.py`

------
Datum izrade plana: 2026-05-15
