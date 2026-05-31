"""Tests for CLI functions in ml_code_smell_detector/cli.py."""

import csv
import os
import textwrap

import pytest

from ml_code_smell_detector.cli import (
    analyze_file,
    build_arg_parser,
    collect_python_files,
    write_csv_report,
    write_txt_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pandas_file(tmp_path):
    """A minimal pandas file that triggers at least one smell."""
    p = tmp_path / "pandas_code.py"
    p.write_text(textwrap.dedent("""\
        import pandas as pd
        df = pd.read_csv('data.csv')
        df.fillna(0, inplace=True)
    """))
    return str(p)


@pytest.fixture
def numpy_file(tmp_path):
    """A minimal numpy file that triggers a smell."""
    p = tmp_path / "numpy_code.py"
    p.write_text(textwrap.dedent("""\
        import numpy as np
        arr = np.array([1, 2, 3])
    """))
    return str(p)


@pytest.fixture
def clean_file(tmp_path):
    """A non-ML file that should produce no smells."""
    p = tmp_path / "plain.py"
    p.write_text("def add(a, b):\n    return a + b\n")
    return str(p)


@pytest.fixture
def sample_results():
    """Pre-built results dict matching the structure returned by analyze_file."""
    return {
        "fake_file.py": {
            "Framework-Specific": [
                {
                    "framework": "Pandas",
                    "name": "InPlace Checker",
                    "fix": "Avoid inplace=True",
                    "benefits": "Cleaner code",
                    "location": "Line 3",
                }
            ],
            "Hugging Face": [],
            "General ML": [
                {
                    "framework": "General ML",
                    "name": "Magic number detected: 0.75",
                    "fix": "Use constants",
                    "benefits": "Readability",
                    "location": "Line 5",
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# analyze_file
# ---------------------------------------------------------------------------

class TestAnalyzeFile:
    def test_returns_dict_with_three_sections(self, pandas_file):
        result = analyze_file(pandas_file)
        assert isinstance(result, dict)
        assert "Framework-Specific" in result
        assert "Hugging Face" in result
        assert "General ML" in result

    def test_framework_section_is_list(self, pandas_file):
        result = analyze_file(pandas_file)
        assert isinstance(result["Framework-Specific"], list)

    def test_non_ml_file_returns_empty_sections(self, clean_file):
        result = analyze_file(clean_file)
        assert result["Framework-Specific"] == []
        assert result["Hugging Face"] == []
        assert result["General ML"] == []

    def test_pandas_file_has_framework_smells(self, pandas_file):
        result = analyze_file(pandas_file)
        assert len(result["Framework-Specific"]) > 0

    def test_smell_entries_have_required_keys(self, pandas_file):
        result = analyze_file(pandas_file)
        for smell in result["Framework-Specific"]:
            for key in ("framework", "name", "fix", "benefits", "location"):
                assert key in smell, f"Missing key '{key}' in smell entry"


# ---------------------------------------------------------------------------
# write_txt_report
# ---------------------------------------------------------------------------

class TestWriteTxtReport:
    def test_creates_txt_file(self, tmp_path, sample_results):
        out = str(tmp_path / "report.txt")
        write_txt_report(sample_results, out)
        assert os.path.exists(out)

    def test_txt_contains_file_path(self, tmp_path, sample_results):
        out = str(tmp_path / "report.txt")
        write_txt_report(sample_results, out)
        content = open(out).read()
        assert "fake_file.py" in content

    def test_txt_contains_smell_name(self, tmp_path, sample_results):
        out = str(tmp_path / "report.txt")
        write_txt_report(sample_results, out)
        content = open(out).read()
        assert "InPlace Checker" in content

    def test_txt_contains_fix(self, tmp_path, sample_results):
        out = str(tmp_path / "report.txt")
        write_txt_report(sample_results, out)
        content = open(out).read()
        assert "Avoid inplace=True" in content

    def test_txt_shows_total_smells(self, tmp_path, sample_results):
        out = str(tmp_path / "report.txt")
        write_txt_report(sample_results, out)
        content = open(out).read()
        assert "Total smells detected" in content


# ---------------------------------------------------------------------------
# write_csv_report
# ---------------------------------------------------------------------------

class TestWriteCsvReport:
    def test_creates_csv_file(self, tmp_path, sample_results):
        out = str(tmp_path / "report.csv")
        write_csv_report(sample_results, out)
        assert os.path.exists(out)

    def test_csv_has_header_row(self, tmp_path, sample_results):
        out = str(tmp_path / "report.csv")
        write_csv_report(sample_results, out)
        with open(out, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "Framework" in header
        assert "Smell/Checker Name" in header
        assert "How to Fix" in header
        assert "Benefits" in header
        assert "File Path" in header
        assert "Location" in header
        assert "Count" in header

    def test_csv_contains_data_row(self, tmp_path, sample_results):
        out = str(tmp_path / "report.csv")
        write_csv_report(sample_results, out)
        with open(out, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) > 1  # header + at least one data row

    def test_csv_data_row_has_correct_values(self, tmp_path, sample_results):
        out = str(tmp_path / "report.csv")
        write_csv_report(sample_results, out)
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        inplace_row = next((r for r in rows if "InPlace Checker" in r["Smell/Checker Name"]), None)
        assert inplace_row is not None
        assert inplace_row["Framework"] == "Pandas"
        assert inplace_row["How to Fix"] == "Avoid inplace=True"


# ---------------------------------------------------------------------------
# collect_python_files
# ---------------------------------------------------------------------------

class TestCollectPythonFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        files = collect_python_files(str(tmp_path), set())
        basenames = [os.path.basename(f) for f in files]
        assert "a.py" in basenames
        assert "b.py" in basenames
        assert "c.txt" not in basenames

    def test_recurses_into_subdirectory(self, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "deep.py").write_text("")
        files = collect_python_files(str(tmp_path), set())
        basenames = [os.path.basename(f) for f in files]
        assert "deep.py" in basenames

    def test_ignores_specified_directories(self, tmp_path):
        ignored = tmp_path / "venv"
        ignored.mkdir()
        (ignored / "ignore_me.py").write_text("")
        (tmp_path / "keep_me.py").write_text("")
        files = collect_python_files(str(tmp_path), {"venv"})
        basenames = [os.path.basename(f) for f in files]
        assert "keep_me.py" in basenames
        assert "ignore_me.py" not in basenames

    def test_returns_empty_for_empty_dir(self, tmp_path):
        files = collect_python_files(str(tmp_path), set())
        assert files == []


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------

class TestBuildArgParser:
    def test_analyze_action_accepted(self):
        parser = build_arg_parser()
        args = parser.parse_args(["analyze", "some/path"])
        assert args.action == "analyze"
        assert args.path == "some/path"

    def test_default_output_dir(self):
        parser = build_arg_parser()
        args = parser.parse_args(["analyze", "."])
        assert args.output_dir == "output"

    def test_custom_output_dir(self):
        parser = build_arg_parser()
        args = parser.parse_args(["analyze", ".", "--output-dir", "reports"])
        assert args.output_dir == "reports"

    def test_ignore_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["analyze", ".", "--ignore", "tests", "docs"])
        assert "tests" in args.ignore
        assert "docs" in args.ignore

    def test_invalid_action_raises(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["deploy", "."])
