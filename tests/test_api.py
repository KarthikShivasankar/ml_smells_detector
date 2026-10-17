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
flag = df["a"] > 0.75
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


class TestSuppressionsAndFilters:
    CODE = """\
import pandas as pd
df = pd.read_csv('a.csv', usecols=['a'])
df.dropna(inplace=True)  # mlscent: ignore[ML.PANDAS.INPLACE_CHECKER]
df.fillna(0, inplace=True)  # mlscent: ignore[ML.NUMPY]
df.ffill(inplace=True)  # mlscent: ignore
"""

    def test_line_suppressions_match_by_prefix(self):
        result = api.analyze_source(self.CODE)
        assert [(f["line"], f["rule_id"]) for f in result["findings"]] == [(4, "ML.PANDAS.INPLACE_CHECKER")]
        assert result["suppressed"] == 2

    def test_file_suppression(self):
        result = api.analyze_source("# mlscent: ignore-file[ML.PANDAS]\n" + self.CODE)
        assert result["findings"] == []

    def test_select_and_ignore(self):
        code = "import numpy as np\nx = np.random.rand(3)\nif x[0] == np.nan:\n    pass\n"
        assert {f["rule_id"] for f in api.analyze_source(code, select=["ML.NUMPY.NAN"])["findings"]} == {
            "ML.NUMPY.NAN_EQUALITY_CHECKER"}
        assert "ML.NUMPY.NAN_EQUALITY_CHECKER" not in {
            f["rule_id"] for f in api.analyze_source(code, ignore=["ML.NUMPY.NAN"])["findings"]}

    def test_syntax_error_reported_once_with_logical_name(self):
        result = api.analyze_source("import numpy as np\ndef broken(:\n", filename="pkg/broken.py")
        assert len(result["errors"]) == 1 and "pkg/broken.py" in result["errors"][0]


class TestAnalyzePaths:
    def test_skips_virtualenvs_and_hidden_dirs(self, tmp_path):
        for d in (".venv", "node_modules", ".hidden", "src"):
            (tmp_path / d).mkdir()
            (tmp_path / d / "m.py").write_text("import numpy as np\nx = np.array([1])\n")
        doc = api.analyze_paths(str(tmp_path))
        assert [f["file"] for f in doc["files"]] == [str(tmp_path / "src" / "m.py")]
        assert doc["summary"]["by_rule"] == {"ML.NUMPY.ARRAY_CREATION_EFFICIENCY": 1}

    def test_missing_path_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            api.analyze_paths(["nope.py"])

    def test_general_findings_carry_fix_text(self):
        findings = api.analyze_source("import pandas as pd\ndf = pd.read_csv('/abs/path/train.csv')\n")["findings"]
        path = next(f for f in findings if f["rule_id"] == "ML.HARDCODED_PATH")
        assert path["how_to_fix"] and path["benefits"]
