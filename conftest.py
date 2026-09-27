"""Root pytest configuration (neutral: shared by tests/spec and tests/builder).

Makes the in-place build importable when pytest is run from the project root without
an editable install.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
