"""TESTER-owned black-box suite for PufferRoyale (SPEC v0.1, docs/SPEC.md).

Every expectation in tests/spec is derived from docs/SPEC.md, the pinned data files in
data/source/, or reference game knowledge -- never from the builder's implementation.

Markers:
  slow -- long-running (sweeps, soak, full matches, ASan build)
  perf -- wall-clock throughput thresholds (SPEC §11)

A missing API must FAIL, never skip: no importorskip anywhere in this directory.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running spec test (sweeps, soak, full matches)")
    config.addinivalue_line("markers", "perf: wall-clock throughput threshold test (SPEC §11)")


@pytest.fixture
def pr():
    """The pufferroyale package. Import failure is a test failure (not a skip)."""
    import pufferroyale  # noqa: F401  -- must exist (SPEC §9, §10)
    return pufferroyale
