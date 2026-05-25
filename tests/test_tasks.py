"""
Smoke tests for tasks.py — ensures the invoke task collection imports
cleanly and surfaces the expected target names.

We deliberately do NOT exercise the task bodies (they shell out to
pytest / uvicorn / kaggle and would loop the test runner on itself).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tasks_module():
    """Import tasks.py without polluting sys.modules between tests."""
    spec = importlib.util.spec_from_file_location("tasks_under_test", REPO_ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tasks_module():
    return _load_tasks_module()


@pytest.fixture(scope="module")
def collection(tasks_module):
    """Build an invoke Collection from the loaded tasks module."""
    from invoke import Collection

    return Collection.from_module(tasks_module)


EXPECTED_TASKS = {
    "install",
    "data",
    "seed",
    "train-classifier",
    "pull-models",
    "eval",
    "api",
    "demo",
    "lint",
    "format",
    "test",
    "security",
    "ci",
    "release-check",
}


def test_tasks_module_imports_cleanly(tasks_module):
    """tasks.py must import without side effects (no shell-out at import time)."""
    assert tasks_module is not None
    assert hasattr(tasks_module, "REPO_ROOT")


def test_expected_invoke_tasks_are_registered(collection):
    """Every documented target in README must exist in the invoke collection."""
    registered = set(collection.task_names.keys())
    missing = EXPECTED_TASKS - registered
    assert not missing, f"Missing invoke targets: {sorted(missing)}"


def test_ci_task_composes_lint_test_security(tasks_module):
    """`inv ci` should call the three quality gates in source order."""
    import inspect

    source = inspect.getsource(tasks_module.ci)
    # Order matters: lint first (fast feedback), then tests, then security scan.
    assert source.index("lint(c)") < source.index("test(c)") < source.index("security(c)")


def test_env_helper_includes_pythonpath(tasks_module):
    """Subprocess tasks must set PYTHONPATH so `src.` imports work."""
    env = tasks_module._env()
    assert "PYTHONPATH" in env
    assert str(tasks_module.REPO_ROOT) in env["PYTHONPATH"]
