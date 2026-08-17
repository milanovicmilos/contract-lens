"""
Task runner for ContractLens — `invoke` (cross-platform replacement for ``make``).

`invoke` is in `requirements-dev.txt`; once installed, every task is one
command:

    inv --list             # show all targets
    inv install            # pip install -r requirements-dev.txt
    inv data               # prepare CUAD datasets (SQuAD + multi-label JSONL)
    inv seed               # seed the RAG corpus into ChromaDB
    inv train-classifier   # local fine-tune (small model — laptop-friendly)
    inv pull-models        # download trained artifacts from Kaggle
    inv eval               # run RAGAS evaluation on N contracts
    inv api                # start the FastAPI server
    inv demo               # parse one PDF end-to-end -> JSON + PDF report
    inv test               # pytest
    inv lint               # black + ruff
    inv security           # bandit
    inv ci                 # everything CI runs (lint + test + security)
    inv release-check      # ci + ensure no uncommitted changes

The single goal: a defence-committee member should be able to clone the
repo and reach an analysed contract in **three commands**:

    inv install
    inv seed
    inv demo --contract CUAD_v1/full_contract_txt/<file>.txt

No shell-specific syntax, no GNU make dependency on Windows, no missing
PYTHONPATH gymnastics — all paths and env handling live in this file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from invoke import task

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # the interpreter that ran `inv`


def _env(extra: dict | None = None) -> dict:
    """Build the env passed to subprocess tasks — always prepends REPO_ROOT to
    PYTHONPATH so ``src.`` imports resolve regardless of the caller's env.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    parts = [str(REPO_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@task(help={"dev": "Install dev requirements (default True). Pass --no-dev for runtime only."})
def install(c, dev=True):
    """Install Python dependencies via pip."""
    req = "requirements-dev.txt" if dev else "requirements.txt"
    c.run(f'"{PYTHON}" -m pip install -r {req}', env=_env())


# ---------------------------------------------------------------------------
# Data + RAG
# ---------------------------------------------------------------------------
@task
def data(c):
    """Build CUAD multi-label + SQuAD JSONL datasets from raw CUAD_v1/."""
    c.run(f'"{PYTHON}" scripts/prepare_squad_dataset.py', env=_env())
    c.run(f'"{PYTHON}" scripts/prepare_multilabel_dataset.py', env=_env())


@task(help={"persist": "Chroma persist dir (default ./chroma_db)"})
def seed(c, persist="./chroma_db"):
    """Seed the legal corpus (GDPR + EU AI Act + practice notes) into ChromaDB."""
    c.run(
        f'"{PYTHON}" scripts/seed_legal_corpus.py --persist-dir {persist}',
        env=_env(),
    )


# ---------------------------------------------------------------------------
# Training (local fallback; Kaggle is the production path)
# ---------------------------------------------------------------------------
@task(
    help={
        "model": "HF model id or local path. Defaults to a small dev model.",
        "epochs": "Training epochs (default 2 — keep small on laptop).",
        "batch": "Per-device batch size.",
    }
)
def train_classifier(c, model="microsoft/deberta-v3-small", epochs=2, batch=8):
    """Fine-tune the classifier locally (small model, short run — laptop sanity)."""
    c.run(
        f'"{PYTHON}" -m src.infrastructure.ai.train_classifier '
        f"--model {model} --epochs {epochs} --batch {batch}",
        env=_env(),
    )


@task
def pull_models(c):
    """Pull Kaggle-trained classifier + extractor artifacts into models/."""
    c.run(f'"{PYTHON}" scripts/pull_kaggle_models.py', env=_env())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@task(
    help={
        "max-contracts": "How many CUAD contracts to score (default 3).",
        "eval-model": "OpenAI model used as the LLM judge.",
        "chroma-dir": "Chroma persist dir for RAG retrieval.",
    }
)
def eval(c, max_contracts=3, eval_model="gpt-4o-mini", chroma_dir="./chroma_db"):
    """Run RAGAS evaluation: faithfulness + relevancy per emitted RiskScore."""
    c.run(
        f'"{PYTHON}" -m src.evaluation.ragas_eval '
        f"--max-contracts {max_contracts} --eval-model {eval_model} "
        f"--chroma-dir {chroma_dir}",
        env=_env(),
    )


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@task(
    help={
        "reload": "Hot-reload on src/ change (default True for dev).",
        "host": "Bind address (default 127.0.0.1; use 0.0.0.0 inside Docker).",
        "port": "TCP port (default 8000).",
    }
)
def api(c, reload=True, host="127.0.0.1", port=8000):
    """Start the FastAPI server (uvicorn)."""
    reload_flag = "--reload" if reload else ""
    c.run(
        f"uvicorn src.api.main:app --host {host} --port {port} {reload_flag}",
        env=_env(),
        pty=False,
    )


@task(help={"contract": "Path to a contract file (.pdf | .docx | .txt | .md)"})
def demo(c, contract):
    """Run scripts/demo_e2e.py: parse -> classify -> RiskScore -> JSON + PDF report."""
    c.run(f'"{PYTHON}" scripts/demo_e2e.py "{contract}"', env=_env())


# ---------------------------------------------------------------------------
# Quality gates (mirror .github/workflows/ci.yml)
# ---------------------------------------------------------------------------
@task
def lint(c):
    """black --check + ruff check on src/ + tests/."""
    c.run(f'"{PYTHON}" -m black --check src tests', env=_env())
    c.run(f'"{PYTHON}" -m ruff check src tests', env=_env())


@task
def format(c):
    """Apply black + ruff --fix to src/ + tests/."""
    c.run(f'"{PYTHON}" -m black src tests', env=_env())
    c.run(f'"{PYTHON}" -m ruff check src tests --fix', env=_env())


@task(help={"cov": "Emit coverage report (default True)."})
def test(c, cov=True):
    """Run pytest."""
    cov_flag = "--cov=src --cov-report=term-missing" if cov else ""
    c.run(f'"{PYTHON}" -m pytest tests/ {cov_flag} -q', env=_env())


@task
def security(c):
    """Bandit scan (config + skip list in pyproject.toml [tool.bandit])."""
    c.run(f'"{PYTHON}" -m bandit -r src -c pyproject.toml -ll', env=_env())


@task
def ci(c):
    """Run everything CI runs: lint + test + security."""
    lint(c)
    test(c)
    security(c)


@task
def release_check(c):
    """CI + assert the working tree is clean (no uncommitted changes)."""
    ci(c)
    result = c.run("git status --porcelain", hide=True, env=_env())
    dirty = result.stdout.strip()
    if dirty:
        print("\nERROR: working tree is dirty. Commit or stash before releasing.")
        print(dirty)
        sys.exit(1)
    print("\nClean working tree — release-check passed.")
