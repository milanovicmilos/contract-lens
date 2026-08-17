# ADR-001 — Clean Architecture + LangGraph for the agent pipeline

**Status:** Accepted (2026-05-22; refactored in PR #2 cleanup)
**Spec mapping:** §3 "System Architecture (The Clean Way)"

## Context

The spec calls for Hexagonal / Clean Architecture so AI models
(infrastructure) can be swapped without touching business logic. The
project also needs a Multi-Agent Collaboration pipeline (§4) with four
distinct stages: Extractor → Validator → Legal Consultant → Risk Auditor.

Two orthogonal questions had to be answered together:

1. **How are the layers separated?** Strict Clean Architecture, looser
   layered approach, or no layering at all?
2. **How is the multi-stage pipeline implemented?** Imperative function
   chain, custom state machine, or an existing agent framework?

## Decision

Adopt **Clean Architecture with four layers** — `domain`, `application`,
`infrastructure`, `api` — with a strict dependency rule (outer layers
depend on inner; inner never depend on outer). Inside the application
layer, implement the four-stage pipeline as a **LangGraph state
machine**.

Concrete placements:
- `src/domain/`: pure-Python value objects (`Contract`, `Clause`,
  `RiskScore`, `RiskPolicy`). No transformers, no FastAPI, no LangGraph imports.
- `src/application/interfaces/`: ports — `IClassifier`, `IExtractor`,
  `ILLMProvider`, `IVectorDatabase`, `IRiskAnalyzer`.
- `src/application/orchestration/orchestrator.py`: the LangGraph state
  machine that wires the four agents. Application layer, not
  infrastructure, because the *composition* of agents is a use case;
  LangGraph itself is a library detail.
- `src/infrastructure/`: concrete adapters — `HFClassifier`,
  `DebertaExtractor`, `OpenAIProvider`, `ChromaWrapper`,
  `pdf_renderer`. Each implements one port from `application/interfaces/`.
- `src/api/`: FastAPI surface. Composes infrastructure into the
  orchestrator at startup.

Bandit + ruff + the test suite enforce that nothing in `domain/` or
`application/` imports from `infrastructure/` or `api/`.

## Consequences

**Wins:**
- Swapping the classifier from DeBERTa to a future legal-LM only
  requires a new `IClassifier` implementation and a one-line env-var
  change. We already exercised this when the classifier was upgraded
  from v6 (no LoRA) to v8 (LoRA) without touching application code.
- Tests can mock at the port boundary. 90 unit tests run in under 17 s
  on a laptop because no test loads HF weights.
- Reviewers can read the orchestrator (300 LOC) end to end without
  pulling in the rest of the codebase — the layer boundary makes the
  dependency surface obvious.

**Costs:**
- Slight indirection: a request takes
  `api/main.py → orchestrator → IClassifier → HFClassifier` instead of
  one straight call. The clarity-vs.-step-count trade-off comes out in
  favour of clarity for a multi-agent system, but a single-model
  service would not pay this cost.
- LangGraph adds a framework dependency. We accept this because
  modelling Extractor → Validator → Consultant → Auditor as an
  imperative function chain would re-implement state-machine wiring
  poorly (see Alternatives).

## Alternatives considered

1. **Pure-Python function chain.** Rejected because the Validator can
   short-circuit the pipeline (header → skip Consultant entirely);
   modelling that as conditional edges in a state machine is cleaner
   than imperative if/return cascades that mix routing with business
   logic.
2. **LangChain LCEL.** Rejected because the spec calls out specifically
   for stateful agent coordination (§4); LCEL is better suited to
   linear prompt-pipe-prompt chains. LangGraph's `StateGraph` matches
   the data-flow we actually have.
3. **No layering / monolithic FastAPI module.** Rejected on
   reviewability and testability grounds. A flat structure works at
   500 LOC; this project crossed that an iteration ago.

## References

- Robert C. Martin, *Clean Architecture: A Craftsman's Guide to
  Software Structure and Design*, 2017.
- LangGraph docs — https://langchain-ai.github.io/langgraph/
- PR #2 (cleanup): moved `RiskScore` into domain and the orchestrator
  out of infrastructure to fix a layering violation that was caught
  during the audit.
