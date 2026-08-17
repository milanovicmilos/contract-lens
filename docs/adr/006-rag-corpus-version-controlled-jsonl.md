# ADR-006 — RAG corpus as version-controlled JSONL, not a vendor knowledge service

**Status:** Accepted (2026-05-25, PR #9)
**Spec mapping:** §3 "Vector DB (ChromaDB/Pinecone): Skladištenje pravnih regulativa (npr. GDPR, EU AI Act 2026) za RAG."

## Context

The first iteration shipped a `scripts/seed_regulations.py` with **12
paraphrased snippets hard-coded as Python literals** inside the seed
script. This was acceptable as a stub for the v1 pipeline but blocked
two things:

1. **Audit / review.** A reviewer can't easily see *which articles* the
   RAG knows about without reading the Python file. Citations in the
   compliance report point at snippets the reader can't independently
   verify.
2. **Retrieval precision.** 12 paragraph-sized snippets give the
   retriever almost no signal to discriminate between related articles
   (e.g., GDPR Art. 28 vs. 32 vs. 33 all touch security / processor
   obligations).

The RAG quality also caps RAGAS faithfulness — docs/RESULTS.md §3 v2
flagged "Path to >0.5 faithfulness" as requiring a real corpus, not
just better justification templates.

## Decision

Store the legal corpus as **version-controlled JSONL files** under
`data/legal_corpus/`:

```
data/legal_corpus/
├── gdpr.jsonl            # 20 GDPR articles
├── eu_ai_act.jsonl       # 13 EU AI Act provisions
└── practice_notes.jsonl  # 15 contract-practice principles
```

Each row is **one article**, with structured metadata:

```json
{
  "id": "gdpr-art-28",                  // stable, idempotent
  "source": "GDPR",
  "article": "28",
  "title": "Processor",
  "topic": "processor",
  "tags": ["dpa","subprocessor","data-processing-agreement"],
  "text": "Processing by a processor shall be governed by a contract that..."
}
```

A loader (`src/data/legal_corpus.py::load_corpus`) reads every
`*.jsonl` under the corpus dir, validates required fields, rejects
duplicate ids, and yields typed `LegalDocument` objects. The seed
script (`scripts/seed_legal_corpus.py`) reads them and **upserts** by
stable id into ChromaDB — so re-running the seed is idempotent and
edits to an article propagate cleanly.

The ChromaWrapper switched from `collection.add()` to
`collection.upsert()` in PR #9 specifically to make this work.

EU primary legislation (GDPR, EU AI Act) entries are paraphrased
article summaries (not verbatim reproductions), freely reusable under
Commission Decision 2011/833/EU. Each entry's `article` field lets a
reviewer cross-check against EUR-Lex.

## Consequences

**Wins:**
- Corpus grew from 12 → **48 article-level entries** with one PR.
  Article-level granularity means RAG retrieval returns a precise
  citation ("GDPR Art. 28") instead of a paraphrased blob.
- Adding a new regulation = add one JSONL row + re-run the seed.
  No code change, no rebuild, no model retrain.
- Diff-able. A reviewer can read `git log -p data/legal_corpus/` to
  see exactly which articles changed when, and why.
- Loader fails loudly on misconfiguration (missing dir, empty dir,
  malformed JSON, duplicate id). No silent fallback to an empty RAG —
  that would mask a real deployment misconfiguration.
- Smoke test at seed-time verifies retrieval: a query for "processor
  obligations" must return GDPR Art. 28 as the top hit. PR #9 captured
  the run output.

**Costs:**
- The corpus is a maintained artifact, not an external knowledge
  service. Updating GDPR (it won't change) or the AI Act (it might —
  delegated acts) requires a PR. For a thesis project this is fine.
  A production deployment would add a periodic ingest job pulling
  from EUR-Lex.
- 48 entries is still tiny vs. a full regulation corpus (GDPR alone
  has 99 articles + 173 recitals). Coverage was prioritised by
  CUAD-category relevance — the articles most likely to be cited
  during contract analysis.

## Alternatives considered

1. **Continue with 12 inline snippets.** Rejected — it was the
   documented bottleneck for both audit and RAGAS faithfulness.
2. **Vendor knowledge service (Pinecone-hosted, Cohere, etc.).**
   Rejected on privacy-first grounds (§1 of spec) — the RAG corpus
   embedding must stay local. ChromaDB on disk satisfies that.
3. **Markdown files per article.** Considered. Rejected because JSONL
   makes the metadata structured and machine-parseable (e.g.,
   `tags`, `article`, `topic` fields), which lets the loader enforce a
   schema. Markdown would either lose the metadata or require
   frontmatter parsing.
4. **A relational DB (Postgres + pgvector) as the corpus store.**
   Rejected — adds infra without changing the audit / authoring
   workflow. The current SQLite + Chroma combination already covers
   both job state and RAG with one local file each.

## References

- PR #9 (`feat(rag): real legal corpus`).
- `src/data/legal_corpus.py` — loader + dataclass + validation.
- `scripts/seed_legal_corpus.py` — seed entry point.
- Commission Decision 2011/833/EU on the reuse of Commission documents.
- [docs/RESULTS.md §3](../RESULTS.md) "Path to >0.5 faithfulness".
