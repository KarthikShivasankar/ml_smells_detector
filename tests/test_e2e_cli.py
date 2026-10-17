"""End-to-end: run the installed CLI as a subprocess, the way an agent or CI job would."""

import json
import subprocess
import sys
from pathlib import Path

from ml_code_smell_detector import api

CORPUS = Path(__file__).parent / "corpus"


def run(*args, stdin=None):
    return subprocess.run([sys.executable, "-m", "ml_code_smell_detector", *args],
                          input=stdin, capture_output=True, text=True, timeout=120)


def test_json_report_over_a_directory():
    proc = run("analyze", str(CORPUS), "--format", "json")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert {"tool", "version", "schema_version", "files", "summary"} <= set(doc)
    assert doc["schema_version"] == api.SCHEMA_VERSION
    assert [f["file"] for f in doc["files"]] == sorted(str(p) for p in CORPUS.glob("*.py"))
    total = sum(len(f["findings"]) for f in doc["files"])
    assert doc["summary"]["findings"] == total == sum(doc["summary"]["by_rule"].values()) > 0
    assert doc["summary"]["errors"] == 0


def test_fail_on_findings_sets_exit_code():
    assert run("analyze", str(CORPUS / "pandas_report.py"), "--format", "json", "--fail-on-findings").returncode == 1
    assert run("analyze", str(CORPUS / "clean_pipeline.py"), "--format", "json", "--fail-on-findings").returncode == 0


def test_missing_path_is_a_usage_error():
    proc = run("analyze", "does/not/exist.py", "--format", "json")
    assert proc.returncode == 2
    assert proc.stdout == ""


def test_stdin_and_rule_selection():
    code = "import pandas as pd\ndf = pd.read_csv('a.csv')\ndf.dropna(inplace=True)\n"
    proc = run("analyze", "-", "--format", "json", "--select", "ML.PANDAS.INPLACE", stdin=code)
    doc = json.loads(proc.stdout)
    assert doc["files"][0]["file"] == "stdin.py"
    assert [(f["line"], f["rule_id"]) for f in doc["files"][0]["findings"]] == [(3, "ML.PANDAS.INPLACE_CHECKER")]


def test_ignore_rules_drops_matching_ids():
    doc = json.loads(run("analyze", str(CORPUS / "pandas_report.py"), "--format", "json",
                         "--ignore-rules", "ML.PANDAS", "ML.MAGIC_NUMBER").stdout)
    assert {f["rule_id"] for f in doc["files"][0]["findings"]} == {"ML.NO_ERROR_HANDLING"}


def test_sarif_output():
    proc = run("analyze", str(CORPUS / "sklearn_train.py"), "--format", "sarif")
    sarif = json.loads(proc.stdout)
    assert sarif["version"] == "2.1.0"
    run_ = sarif["runs"][0]
    results = run_["results"]
    assert results and all(r["ruleId"] == run_["tool"]["driver"]["rules"][r["ruleIndex"]]["id"] for r in results)
    lines = [r["locations"][0]["physicalLocation"].get("region", {}).get("startLine") for r in results]
    assert 14 in lines  # the data leakage fit_transform


def test_rules_and_version():
    rules = json.loads(run("rules").stdout)
    assert len({r["rule_id"] for r in rules}) == len(rules)
    assert all(r["how_to_fix"] for r in rules if r["rule_id"] != "ML.OTHER")
    assert run("--version").stdout.strip().endswith(api.__version__)


def test_text_report_honours_suppressions(tmp_path):
    proc = run("analyze", str(CORPUS / "suppressed.py"), "--output-dir", str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    report = (tmp_path / "analysis_report.txt").read_text()
    assert "Chain Indexing" in report and "InPlace Checker" not in report
