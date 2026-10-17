"""End-to-end corpus: every file under tests/corpus/ declares the exact findings it must produce.

    sales.dropna(inplace=True)  # expect: ML.PANDAS.INPLACE_CHECKER
    # expect-file: ML.PYTORCH.LOGGING_CHECKER          <- a file-level (line 0) finding

A missing finding is a recall regression, an extra one a precision regression. After a deliberate rule change,
review the new output and update the annotations. Add a corpus file for every new rule or fixed false positive.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

from ml_code_smell_detector import api

CORPUS = Path(__file__).parent / "corpus"
EXPECT = re.compile(r"#\s*expect(-file)?:\s*(.+)$")


def expected_findings(path):
    expected = Counter()
    for lineno, text in enumerate(path.read_text().splitlines(), 1):
        match = EXPECT.search(text)
        if match:
            for rule_id in match.group(2).split(","):
                expected[(0 if match.group(1) else lineno, rule_id.strip())] += 1
    return expected


@pytest.mark.parametrize("path", sorted(CORPUS.glob("*.py")), ids=lambda p: p.name)
def test_corpus_file(path):
    result = api.analyze_file(str(path))
    assert result["errors"] == []
    actual = Counter((f["line"], f["rule_id"]) for f in result["findings"])
    expected = expected_findings(path)
    assert sorted((actual - expected).elements()) == [], "unexpected findings (precision regression)"
    assert sorted((expected - actual).elements()) == [], "missing findings (recall regression)"


def test_clean_file_has_no_findings():
    assert api.analyze_file(str(CORPUS / "clean_pipeline.py"))["findings"] == []


def test_corpus_rule_ids_are_in_the_catalogue():
    catalogue = {r["rule_id"] for r in api.list_rules()}
    used = {rule_id for path in CORPUS.glob("*.py") for _, rule_id in expected_findings(path)}
    assert used <= catalogue
