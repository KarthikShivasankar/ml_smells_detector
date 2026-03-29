# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ml_code_smell_detector` is a Python CLI tool that statically analyzes Python ML code using AST (via `astroid`) to detect code smells across ML frameworks (Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, Hugging Face Transformers).

## Commands

### Installation (development)
```bash
uv pip install -e ".[dev]"
```

### Run the tool
```bash
ml_smell_detector analyze <path_to_file_or_directory> [--output-dir output]
```
Outputs `analysis_report.txt` and `analysis_report.csv` to the specified output directory.

### Run tests
```bash
uv run pytest tests/
# Single test file:
uv run pytest tests/test_framework_detector.py
# Single test:
uv run pytest tests/test_sklearn_cv_detector.py::TestCrossValidationDetector::test_all_cv_files
```

### Build documentation
```bash
# Windows:
rebuild_docs.bat
# Manual:
cd docs && sphinx-build -b html source build
```

## Architecture

### Entry point
- `ml_code_smell_detector/cli.py` — `main()` parses CLI args, walks files, calls `analyze_file()`, writes reports
- Console script: `ml_smell_detector` → `ml_code_smell_detector.cli:main`

### Detector classes (in `ml_code_smell_detector/detectors/`)
All three detectors share the same interface:
- `detect_smells(file_path)` — parses the file with `astroid.parse()` and runs checks
- `get_results()` — returns list of dicts with keys: `name`, `framework`, `fix`, `benefits`, `location`
- `generate_report()` — returns formatted string

| Class | File | Scope |
|---|---|---|
| `FrameworkSpecificSmellDetector` | `framework_detector.py` | Pandas, NumPy, Sklearn, PyTorch, TensorFlow |
| `HuggingFaceSmellDetector` | `huggingface_detector.py` | Hugging Face Transformers |
| `ML_SmellDetector` | `ml_detector.py` | General ML practices |

### Detection pattern
Each detector's `visit_module()` method calls individual `detect_*()` / `check_*()` methods that walk the AST and append smell dicts to `self.smells`. Framework presence is detected by inspecting imports via `get_imported_modules()` from `utils.py`.

### Utilities (`ml_code_smell_detector/utils.py`)
AST helper functions built on `astroid`: extracting imports, function/class/variable names, call names, attribute access, and constant values.

## Package Configuration

- Package metadata: `pyproject.toml` (build backend: hatchling)
- Runtime dependencies: `astroid`, `tqdm` (the tool itself does not require ML frameworks installed)
- Dev dependencies: `pytest`, `pytest-cov`, `flake8` — install with `uv pip install -e ".[dev]"`
- Build: `uv build` → `dist/`
- Publish: `uv publish --token pypi-<token>`

