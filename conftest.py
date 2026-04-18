"""Root conftest for pytest.

Prepends the project root to sys.path so tests can import project modules
(e.g., `from scripts.numeric_sanity import ...`) without installing the
project as a package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
