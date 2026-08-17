# ADR-003 — OpenAI as default LLM, behind an `ILLMProvider` Strategy port

**Status:** Accepted (2026-05-22)
**Spec mapping:** §4 "Legal Consultant Agent (Koristi OpenAI API)", §5 "Strategy Pattern za LLM provajdere"

## Context

The spec calls for the Legal Consultant agent to use OpenAI's GPT-4o
class model for clause analysis, with the explicit caveat: "OpenAI
tokeni se koriste isključivo za anonimizovane tekstove ili specifične
složene procene" (§4) — tokens are only spent on validated, anonymised
clauses, not the whole contract. §5 also mandates a Strategy Pattern
for LLM providers.

Operational realities:
- Per-clause GPT-4o-mini calls cost ~$0.0001 input + $0.0006 output.
  At 50 chunks per contract, that's ~$0.03 per contract.
- Outages and rate limits happen. The pipeline must not become unusable
  when OpenAI is degraded.
- Future deployments may need to swap to Anthropic, Azure OpenAI, or
  a local model — for cost, jurisdiction, or sovereignty reasons.

## Decision

Define an `ILLMProvider` Strategy port in
`src/application/interfaces/illm_provider.py`:

```python
class ILLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: List[LLMMessage], *,
             temperature: float = 0.0,
             max_tokens: Optional[int] = None,
             response_format: Optional[str] = None) -> LLMResponse: ...
```

Ship **one** concrete implementation today —
`src/infrastructure/llm/openai_provider.py` using `openai>=1.0`
directly (no LangChain wrapper on the production path; see PR #6 for
the removal). Default model: `gpt-4o-mini` because it sits at the
best price/quality point for our two workloads — Consultant clause
analysis and RAGAS faithfulness judging. Override with
`OPENAI_MODEL=gpt-4o` when higher quality justifies the cost.

Graceful degradation: the orchestrator accepts `llm_provider=None`
without erroring. When LLM is unavailable (no key, network down, or
`DISABLE_LLM=1` set), the Consultant step returns a rule-based note
and the rest of the pipeline still produces RiskScores from the
RiskPolicy. **Tested** in `tests/test_orchestrator.py::test_orchestrator_valid_flow_with_extractor`.

## Consequences

**Wins:**
- Swapping to Anthropic Claude would be one new class implementing
  `ILLMProvider.chat` and one env var. The orchestrator, evaluator,
  and tests stay untouched. Same for Azure OpenAI, local llama.cpp,
  vLLM, or a sovereign EU host.
- LangChain dependency removed (PR #6). The Strategy port is a thin
  300-LOC adapter; LangChain added a transitive surface we didn't use.
- RAGAS evaluator (`LLMEvaluator`) takes the same port, so the
  Consultant and the LLM-as-judge can be powered by different models
  (e.g., Consultant on `gpt-4o-mini`, judge on `gpt-4o` for stricter
  scoring) without code changes.

**Costs:**
- Adopting the Strategy adds one indirection vs. calling `openai.chat`
  directly. The cost is one extra `LLMMessage` dataclass and one
  `chat()` method on the port — trivial.
- The deferred-LLM rule-based fallback produces lower-faithfulness
  justifications (see [docs/RESULTS.md §3 v2 metrics](../RESULTS.md)).
  The fallback is correct (RiskScores still emit), but the
  faithfulness score drops because the Consultant's grounded analysis
  is what makes the justification verifiable. Operators who care about
  faithfulness must wire a real LLM provider.

## Alternatives considered

1. **LangChain `ChatModel` interface.** Initially used. Removed in
   PR #6 — the dual-mode (LangChain `.invoke()` + ILLMProvider
   `.chat()`) added branching to the orchestrator and evaluator
   without callers actually using LangChain. The Strategy port we kept
   is vendor-neutral too; we just stopped paying the LangChain
   abstraction tax on top of it.
2. **Hard-code OpenAI.** Rejected on spec grounds (§5) and on real
   vendor-swap risk: the EU AI Act in particular adds compliance
   obligations on AI service providers, and a future deployment may
   need to route to a different vendor for jurisdiction reasons.
3. **Local LLM only.** Rejected because per-clause legal reasoning
   needs a frontier-class model. A 7-B local model produces
   reasonable summaries but not citation-grade legal analysis at
   acceptable cost / latency on our hardware. Deferred to v2 of the
   project as a `LocalLLMProvider` implementation behind the same port.

## References

- *Design Patterns* (Gamma et al., 1994) — Strategy.
- OpenAI Python SDK docs — https://github.com/openai/openai-python.
- PR #6 (`refactor: drop langchain runtime dependency`).
