"""
Batch RAGAS-style evaluation harness.

Runs the orchestrator over a held-out set of contracts, then scores every
produced RiskScore with two metrics:
- Faithfulness: is the justification grounded in the source clause text?
- Relevancy: does the extracted span actually relate to the category?

Outputs:
- evaluation_report.json: aggregate + per-category metrics
- evaluation_details.jsonl: one row per RiskScore for drill-down

USAGE:
    python -m src.evaluation.ragas_eval \
        --input data/processed/cuad_squad.jsonl \
        --output docs/evaluation_report.json \
        --max-contracts 25 \
        --model gpt-4o-mini

If OPENAI_API_KEY is missing or no LLM client is available, the evaluator
will skip LLM calls and produce only the dataset statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.application.evaluation.evaluator import EvaluationResult, LLMEvaluator
from src.application.interfaces.irisk_analyzer import RiskScore
from src.domain.risk_policy import RiskPolicy
from src.infrastructure.agents.orchestrator import ContractOrchestrator
from src.infrastructure.ai.hf_classifier import HFClassifier
from src.infrastructure.database.chroma_wrapper import ChromaWrapper

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_llm_provider(model: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from src.infrastructure.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model, api_key=api_key)
    except Exception as exc:
        logger.warning(f"Could not build OpenAIProvider: {exc}")
        return None


def _load_contracts(input_path: Path, max_contracts: int) -> List[Dict[str, Any]]:
    """Load unique (contract_id, context) pairs from a CUAD SQuAD JSONL."""
    seen: Dict[str, Dict[str, Any]] = {}
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("contract_id", row.get("title", ""))
            if cid and cid not in seen:
                seen[cid] = {"contract_id": cid, "context": row.get("context", "")}
            if len(seen) >= max_contracts:
                break
    return list(seen.values())


def _aggregate(scores: List[EvaluationResult]) -> Dict[str, float]:
    if not scores:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    values = [s.score for s in scores]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate(
    input_path: Path,
    output_path: Path,
    classifier_model: str,
    eval_model: str,
    max_contracts: int,
    chunk_size: int,
    chroma_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full evaluation pipeline and write report; return the aggregate dict."""
    logger.info(f"Loading first {max_contracts} contracts from {input_path}")
    contracts = _load_contracts(input_path, max_contracts)
    logger.info(f"Loaded {len(contracts)} contracts")

    llm_provider = _build_llm_provider(eval_model)
    if llm_provider is None:
        logger.warning("No LLM provider available; RAGAS metrics will be skipped.")
        evaluator = None
    else:
        evaluator = LLMEvaluator(llm_provider=llm_provider)

    classifier = HFClassifier(model_name_or_path=classifier_model)
    vector_db = (
        ChromaWrapper(persist_directory=chroma_dir, collection_name="legal_regulations")
        if chroma_dir
        else None
    )
    orchestrator = ContractOrchestrator(
        classifier=classifier,
        risk_policy=RiskPolicy(),
        extractor=None,
        llm_provider=llm_provider,
        vector_db=vector_db,
    )

    all_risks: List[RiskScore] = []
    faithfulness_per_cat: Dict[str, List[EvaluationResult]] = defaultdict(list)
    relevancy_per_cat: Dict[str, List[EvaluationResult]] = defaultdict(list)
    details_path = output_path.with_suffix(".details.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(details_path, "w", encoding="utf-8") as details_file:
        for idx, contract in enumerate(contracts, 1):
            cid = contract["contract_id"]
            ctx = contract["context"]
            if not ctx.strip():
                continue
            logger.info(f"[{idx}/{len(contracts)}] Analyzing {cid} ({len(ctx)} chars)")

            for start in range(0, len(ctx), chunk_size):
                chunk = ctx[start : start + chunk_size]
                if len(chunk.split()) < 20:
                    continue

                try:
                    risks = orchestrator.analyze(chunk, source_doc=cid)
                except Exception as exc:
                    logger.warning(f"Orchestrator failed on {cid} chunk {start}: {exc}")
                    continue

                for risk in risks:
                    all_risks.append(risk)
                    record: Dict[str, Any] = {
                        "contract_id": cid,
                        "category": risk.category,
                        "risk_level": risk.risk_level,
                        "policy_score": risk.score,
                        "extracted_span": risk.extracted_span[:300],
                    }
                    if evaluator is not None:
                        try:
                            f_res = evaluator.evaluate_faithfulness(risk, chunk)
                            r_res = evaluator.evaluate_relevancy(risk.category, risk.extracted_span)
                            faithfulness_per_cat[risk.category].append(f_res)
                            relevancy_per_cat[risk.category].append(r_res)
                            record["faithfulness"] = f_res.score
                            record["relevancy"] = r_res.score
                        except Exception as exc:
                            logger.warning(f"LLM judge failed: {exc}")
                    details_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_faith = [s for lst in faithfulness_per_cat.values() for s in lst]
    all_rel = [s for lst in relevancy_per_cat.values() for s in lst]

    report: Dict[str, Any] = {
        "config": {
            "classifier_model": classifier_model,
            "eval_model": eval_model,
            "max_contracts": max_contracts,
            "chunk_size": chunk_size,
            "n_contracts_processed": len(contracts),
            "n_risks_emitted": len(all_risks),
            "llm_judge_enabled": evaluator is not None,
        },
        "aggregate": {
            "faithfulness": _aggregate(all_faith),
            "relevancy": _aggregate(all_rel),
        },
        "per_category": {
            cat: {
                "faithfulness": _aggregate(faithfulness_per_cat.get(cat, [])),
                "relevancy": _aggregate(relevancy_per_cat.get(cat, [])),
            }
            for cat in sorted(set(faithfulness_per_cat) | set(relevancy_per_cat))
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Aggregate report written to {output_path}")
    logger.info(f"Per-risk details written to {details_path}")

    if all_faith:
        logger.info(
            f"FINAL: faithfulness mean={report['aggregate']['faithfulness']['mean']:.3f}, "
            f"relevancy mean={report['aggregate']['relevancy']['mean']:.3f} "
            f"(n_risks={len(all_risks)})"
        )
    else:
        logger.info(f"FINAL: {len(all_risks)} risks emitted (no LLM judge metrics).")
    return report


def main():
    parser = argparse.ArgumentParser(description="RAGAS-style evaluation over CUAD contracts")
    parser.add_argument("--input", default="data/processed/cuad_squad.jsonl")
    parser.add_argument("--output", default="docs/evaluation_report.json")
    parser.add_argument(
        "--classifier-model",
        default=os.getenv("CLASSIFIER_MODEL", "prajjwal1/bert-tiny"),
    )
    parser.add_argument("--eval-model", default="gpt-4o-mini")
    parser.add_argument("--max-contracts", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument(
        "--chroma-dir",
        default=os.getenv("CHROMA_DIR"),
        help="If set, RAG retrieval is enabled during evaluation.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    evaluate(
        input_path=root / args.input,
        output_path=root / args.output,
        classifier_model=args.classifier_model,
        eval_model=args.eval_model,
        max_contracts=args.max_contracts,
        chunk_size=args.chunk_size,
        chroma_dir=args.chroma_dir,
    )


if __name__ == "__main__":
    main()
