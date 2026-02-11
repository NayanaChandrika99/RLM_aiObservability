# ABOUTME: Ensures local package imports resolve from the repository root during pytest runs.
# ABOUTME: Prevents environment-dependent import resolution differences in local and CI execution.

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep Phoenix metadata under the repository so unit tests do not depend on home-directory permissions.
os.environ.setdefault("PHOENIX_WORKING_DIR", str(REPO_ROOT / ".phoenix_data"))
