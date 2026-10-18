"""Release metadata agrees: pyproject version, installed package, docs and changelog (publish.yml checks the tag)."""

import re
from pathlib import Path

from ml_code_smell_detector import api

ROOT = Path(__file__).parent.parent
VERSION = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)


def test_installed_package_reports_the_pyproject_version():
    # A stale editable install reports an old version: re-run `uv sync` after bumping
    assert api.__version__ == VERSION


def test_changelog_has_a_section_for_the_version():
    changelog = (ROOT / "docs" / "source" / "changelog.rst").read_text()
    assert re.search(rf"^{re.escape(VERSION)} \(\d{{4}}(-\d\d-\d\d)?\)\n-+\n", changelog, re.M), \
        f"docs/source/changelog.rst needs a '{VERSION} (YYYY-MM-DD)' section"


def test_docs_read_the_version_from_pyproject():
    assert "release = '" not in (ROOT / "docs" / "source" / "conf.py").read_text()


def test_workflows_reference_existing_jobs_and_files():
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text()
    assert "uses: ./.github/workflows/ci.yml" in publish
    assert "workflow_call:" in (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert (ROOT / "tests" / "corpus").is_dir()  # the wheel smoke test analyses it
