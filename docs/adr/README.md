# Architecture Decision Records (ADRs)

This directory captures the **why** behind ContractLens's load-bearing
architectural choices, in the format proposed by Michael Nygard
(*Documenting Architecture Decisions*, 2011). Each ADR is a short
markdown file with five sections: Status, Context, Decision,
Consequences, Alternatives.

Why ADRs in this thesis project? Two readers:

1. **A reviewer / committee member** picking up the repo after the
   defense. They need to know not just *what* the system does — that's
   in `docs/arch.md` — but *which trade-offs were deliberately taken*
   and *which were left for future work*.
2. **A future maintainer** (or future-me) considering a change. Reading
   the relevant ADR before touching the code surfaces the constraint
   that originally drove the design, so the change either respects it
   or knowingly supersedes it.

## Index

| # | Title | Status |
|---|---|---|
| [ADR-001](001-clean-architecture-with-langgraph.md) | Clean Architecture + LangGraph for the agent pipeline | Accepted |
| [ADR-002](002-lora-on-deberta-base-not-full-fine-tune.md) | LoRA on DeBERTa-v3-base instead of full fine-tuning or -large | Accepted |
| [ADR-003](003-openai-as-default-with-strategy-port.md) | OpenAI as default LLM, behind an ILLMProvider Strategy port | Accepted |
| [ADR-004](004-sqlite-job-store-not-celery.md) | SQLite-backed job store instead of Celery + Redis | Accepted (with documented scaling cliff) |
| [ADR-005](005-traceability-via-spans-not-rule-only.md) | Risk traceability via extractor spans + verbatim quote, not rule-only | Accepted |
| [ADR-006](006-rag-corpus-version-controlled-jsonl.md) | RAG corpus as version-controlled JSONL, not a vendor knowledge service | Accepted |

## Process

A new ADR is created when a change:
- Is hard to reverse later (vendor lock-in, schema change, framework swap).
- Forces a measurable trade-off (cost vs. latency, recall vs. precision, simplicity vs. scale).
- Closes off an alternative the reader would naturally ask about.

Editorial rules: numbered sequentially, never renumbered. **Superseded**
ADRs stay in the index with their final status so the historical
reasoning chain remains intact.
