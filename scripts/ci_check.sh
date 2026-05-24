#!/usr/bin/env bash
# CI parity check: run the exact same commands that .github/workflows/ci.yml
# executes, in a Python 3.11 venv that matches the GitHub Actions runner.
# This MUST be run (and pass) before every push to a branch with an open PR,
# or CI will fail with the same kind of import / format / lint errors as
# happened on 2026-05-24 when pypdf / python-docx / reportlab / langgraph
# were not declared in requirements-dev.txt.
#
# USAGE:
#   bash scripts/ci_check.sh           # full reproduction (first run ~5-10 min)
#   bash scripts/ci_check.sh --recreate   # rebuild the venv from scratch
#
# Requires: Python 3.11 on PATH (`py -3.11` on Windows, `python3.11` on Linux/macOS).
set -euo pipefail

VENV_DIR=".venv-ci-check"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

RECREATE=0
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# 1. Locate a Python 3.11 interpreter.
if command -v py >/dev/null 2>&1; then
  PYTHON_311=(py -3.11)
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_311=(python3.11)
else
  echo "Python 3.11 not found. Install it before running CI parity check." >&2
  exit 2
fi

# 2. Create / reuse the dedicated CI venv.
if [ "$RECREATE" = "1" ] && [ -d "$VENV_DIR" ]; then
  echo "--- removing existing $VENV_DIR ---"
  rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "--- creating $VENV_DIR with Python 3.11 ---"
  "${PYTHON_311[@]}" -m venv "$VENV_DIR"
fi

# 3. Locate the venv python (Windows: Scripts/, *nix: bin/).
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
  VPY="$VENV_DIR/Scripts/python.exe"
elif [ -x "$VENV_DIR/bin/python" ]; then
  VPY="$VENV_DIR/bin/python"
else
  echo "Cannot find venv python in $VENV_DIR" >&2
  exit 2
fi

echo "--- venv python: $($VPY --version) ---"

# 4. Install requirements (cached after first run).
echo "--- installing requirements-dev.txt (may take several minutes on first run) ---"
"$VPY" -m pip install --quiet --upgrade pip
"$VPY" -m pip install --quiet --prefer-binary -r requirements-dev.txt

# 5. The three CI checks, run in the same order as .github/workflows/ci.yml.
echo
echo "===== black --check ====="
"$VPY" -m black --check src tests

echo
echo "===== ruff check ====="
"$VPY" -m ruff check src tests

echo
echo "===== pytest with coverage ====="
"$VPY" -m pytest tests/ --cov=src --cov-report=term-missing -q

echo
echo "===== CI parity check PASSED ====="
