"""Tests for the machine-friendly API and the JSON CLI (ml_code_smell_detector.api)."""

import json
import sys

from ml_code_smell_detector import api, cli

FINDING_KEYS = {"rule_id", "detector", "framework", "name", "message", "line", "snippet", "how_to_fix", "benefits"}

SOURCE = """\
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.DataFrame({"a": [1, 2]})
df["a"][0] = 5
threshold = 0.75
X_train, X_test = train_test_split([[0], [1]])
"""


class TestAnalyzeSource:
    def test_unified_schema_and_stable_ids(self):
        result = api.analyze_source(SOURCE)
        assert result["file"] == "module.py"
        assert result["errors"] == []
        assert all(set(f) == FINDING_KEYS for f in result["findings"])
        ids = {f["rule_id"] for f in result["findings"]}
        assert {"ML.PANDAS.CHAIN_INDEXING", "ML.SCIKITLEARN.RANDOMNESS_CONTROL_CHECKER", "ML.MAGIC_NUMBER"} <= ids
        lines = [(f["line"], f["rule_id"]) for f in result["findings"]]
        assert lines == sorted(lines)

    def test_detector_errors_are_returned_not_printed(self, capsys):
        result = api.analyze_source("import numpy as np\ndef broken(:\n")
        assert result["errors"]
        assert capsys.readouterr().err == ""

    def test_every_emitted_rule_id_is_listed(self):
        listed = {r["rule_id"] for r in api.list_rules()}
        assert len(listed) == len(api.list_rules())
        assert {f["rule_id"] for f in api.analyze_source(SOURCE)["findings"]} <= listed


class TestJsonCli:
    def test_analyze_json_prints_document_to_stdout(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "module.py"
        path.write_text(SOURCE)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["ml_smell_detector", "analyze", str(path), "--format", "json"])
        cli.main()
        doc = json.loads(capsys.readouterr().out)
        assert doc["files"][0]["file"] == str(path)
        assert any(f["rule_id"] == "ML.PANDAS.CHAIN_INDEXING" for f in doc["files"][0]["findings"])
        assert not (tmp_path / "output").exists()

    def test_rules_action_prints_catalogue(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["ml_smell_detector", "rules"])
        cli.main()
        assert {"rule_id", "name"} <= set(json.loads(capsys.readouterr().out)[0])
