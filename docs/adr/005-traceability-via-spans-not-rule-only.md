# ADR-005 — Risk traceability via extractor spans + verbatim quote, not rule-only

**Status:** Accepted (2026-05-21; refined in PR #9 RAG-corpus work)
**Spec mapping:** §1 "Transparentnost: Svaki Risk Score mora biti potkrepljen citatom iz ugovora i referencom na važeći zakon."

## Context

The Transparency requirement is the project's main differentiator vs.
the "black box" legal-AI SaaS landscape. Every emitted `RiskScore` must
carry enough provenance for a legal reviewer to verify the model's
reasoning against the source document.

Concretely, a faithful RiskScore needs to answer:
1. **Where** in the contract did the model detect this risk?
   (Character offsets into the source document.)
2. **What text** did it cite?
   (The verbatim quote, not a paraphrase.)
3. **Which law / principle** does it map to?
   (RAG-retrieved article reference.)
4. **What rule** triggered the classification?
   (The RiskPolicy keyword + rationale.)

RAGAS faithfulness measurements (docs/RESULTS.md §3) showed the v1
implementation — generic rule-based templates with no in-line citation
— hit a mean faithfulness of **0.185**. The v2 implementation embedded
the *triggering keyword + offset* verbatim into every justification and
lifted faithfulness to **0.231 (+25 %)**.

## Decision

Pin the `RiskScore` value object in `src/domain/risk_score.py` with
four traceability fields:

```python
@dataclass
class RiskScore:
    category: str
    risk_level: str
    score: float
    justification: str          # human-readable, must cite verbatim
    extracted_span: str         # what the model considers the clause
    metadata: Dict[str, Any]    # classifier confidence, RAG hits, ...
    span_start_offset: Optional[int] = None   # absolute char offset in source
    span_end_offset: Optional[int] = None
    source_doc: Optional[str] = None          # filename or contract id
```

Every layer that produces a `RiskScore` is obligated to populate the
offsets when an extractor is available. The orchestrator wires the
extractor's output into `span_start_offset` / `span_end_offset`; the
PDF report renderer prints "(offset X-Y)" next to every risk; the JSON
report serialises them verbatim.

The `RiskPolicy._quote_around()` helper builds a short verbatim quote
around the triggering keyword and embeds it into the justification
text. This is what carried the v1 → v2 faithfulness lift.

## Consequences

**Wins:**
- A reviewer can `grep` the source document for the quoted text and
  verify the offset is correct in under 10 seconds per RiskScore.
- The traceability fields survive serialisation through three formats
  (JSON, PDF, the SQLite job-store result column) — see `test_compliance_report.py`
  and `test_api_upload.py` for end-to-end assertions.
- RAGAS faithfulness improved without any model change — the
  measurement (docs/RESULTS.md §3, v1 → v2) proves the citation
  embedding alone delivered +25 % mean / +57 % median faithfulness on
  the same 3 contracts and the same classifier.

**Costs (honest):**
- When the extractor is disabled (default until the v8 fine-tune from
  PR #4 ships an evaluation-grade model), `span_start_offset` and
  `span_end_offset` are `None` and the entire chunk is treated as the
  span. That weakens the traceability claim — the reviewer can verify
  the clause exists, but not the exact span.
- Faithfulness ceiling at 0.231 mean reflects the rule-based
  justification template's intrinsic limit. The next step (LLM-rewritten
  justifications, PR-G in the roadmap) replaces the template with an
  LLM rewrite grounded in the extractor span — projected ceiling
  per docs/RESULTS.md "Path to >0.5 faithfulness" is ~0.50.

## Alternatives considered

1. **Rule-based justifications only, no extractor spans.** What v1
   shipped. Faithfulness 0.185 mean — failed the Transparency goal.
2. **LLM-generated justifications only.** Would produce fluent but
   potentially hallucinated text. Rejected because the spec demands
   *verifiable* citations, not plausible-sounding ones.
3. **Drop offsets, keep only the verbatim quote.** Rejected on review
   ergonomics — a 100-page contract may contain a similar clause in
   multiple places. Offsets disambiguate.

## References

- PR #2 (cleanup): moved `RiskScore` into `domain/` so the traceability
  contract is enforced at the Clean Architecture boundary, not in a
  vendor adapter.
- [docs/RESULTS.md §3](../RESULTS.md) — RAGAS v1 → v2 lift table.
- `src/domain/risk_policy.py::RiskPolicy._quote_around` — the helper
  that builds the verbatim citation.
- `src/infrastructure/reporting/pdf_renderer.py` — where the offset
  pair is rendered in the human-readable compliance report.
