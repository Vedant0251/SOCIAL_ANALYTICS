"""Pytest Root Configuration.

Ensures project root is added to sys.path across all test executions.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
