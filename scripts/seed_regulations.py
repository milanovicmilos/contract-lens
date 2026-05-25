"""
DEPRECATED: thin shim that forwards to scripts/seed_legal_corpus.py.

The original implementation hard-coded 12 paraphrased snippets in this
file; the corpus is now structured per article under data/legal_corpus/.
This shim preserves the old entrypoint so existing CI / Docker scripts
continue to work, but logs a deprecation warning so callers migrate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.seed_legal_corpus import main as _real_main  # noqa: E402

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.warning(
        "scripts/seed_regulations.py is DEPRECATED. "
        "Run scripts/seed_legal_corpus.py instead; this shim will be removed "
        "in a future release."
    )
    _real_main()
