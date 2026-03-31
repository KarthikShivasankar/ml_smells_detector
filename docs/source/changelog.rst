Changelog
=========

0.1.0 (2024)
------------

- Initial release.
- Framework-specific smell detector covering Pandas, NumPy, Scikit-learn, TensorFlow, and PyTorch.
- Hugging Face Transformers smell detector.
- General ML smell detector.
- CLI entry point ``ml_smell_detector analyze``.
- Output as ``analysis_report.txt`` and ``analysis_report.csv``.
- Migrated package configuration to ``pyproject.toml`` with hatchling build backend.
- Fixed astroid module caching by using file basename as module name.
