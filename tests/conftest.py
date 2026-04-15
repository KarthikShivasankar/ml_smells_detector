"""Shared pytest fixtures for ml_code_smell_detector tests."""

import textwrap
import pytest


@pytest.fixture
def tmp_py(tmp_path):
    """Return a helper that writes dedented Python code to a temp .py file.

    Usage::

        def test_foo(tmp_py):
            path = tmp_py("import pandas as pd\\ndf.iterrows()")
            ...
    """
    def _make(code: str, filename: str = "test_code.py") -> str:
        p = tmp_path / filename
        p.write_text(textwrap.dedent(code))
        return str(p)

    return _make


def smell_names(smells):
    """Extract the 'name' field from a list of smell dicts (framework detector)."""
    return [s["name"] for s in smells]


def smell_descriptions(smells):
    """Extract the 'smell' field from a list of smell dicts (ml/hf detectors)."""
    return [s["smell"] for s in smells]
