Changelog
=========

0.1.2 (2026)
------------

- Fixed a ``SyntaxError`` on Python 3.10/3.11: ``ML_SmellDetector`` used a
  multi-line f-string expression (PEP 701) that is only valid on Python 3.12+,
  which prevented the module from importing on the lower supported versions.
- Added `Ruff <https://docs.astral.sh/ruff/>`_ as the primary linter and import
  sorter (configured in ``pyproject.toml``); both ruff and flake8 run clean.
- Added GitHub Actions CI (lint, a Python 3.10-3.13 test matrix, and a
  build + ``twine check``).
- Added a PyPI Trusted Publishing (OIDC) release workflow.
- Added ``AGENTS.md`` and stopped tracking generated ``docs/build`` artifacts.

0.1.1 (2026)
------------

- Migrated package configuration to ``pyproject.toml`` with the hatchling build backend.
- Fixed astroid module caching by using the file basename as the module name.
- Updated node-type imports to ``astroid.nodes`` to remove deprecation warnings and stay forward-compatible with astroid v5.
- Added a ``.flake8`` configuration so the 150-character line limit is honored by linters and CI.
- Resolved security advisories in transitive dependencies and added a citation to the README.
- Expanded and corrected documentation (installation, usage, and packaging metadata).

0.1.0 (2024)
------------

- Initial release.
- Framework-specific smell detector covering Pandas, NumPy, Scikit-learn, TensorFlow, and PyTorch.
- Hugging Face Transformers smell detector.
- General ML smell detector.
- CLI entry point ``ml_smell_detector analyze``.
- Output as ``analysis_report.txt`` and ``analysis_report.csv``.
