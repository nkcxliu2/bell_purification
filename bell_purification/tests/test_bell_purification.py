"""
Unit and regression test for the bell_purification package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import bell_purification


def test_bell_purification_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bell_purification" in sys.modules
