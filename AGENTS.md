# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

`ml-code-smell-detector` is a Python CLI that statically analyzes Python ML code
using AST (via `astroid`) to detect code smells across Pandas, NumPy,
Scikit-learn, PyTorch, TensorFlow, and Hugging Face Transformers. It performs
static analysis only and does **not** require any ML framework to be installed.

- Package: `ml_code_smell_detector/`
- Distribution name (PyPI): `ml-code-smell-detector`
- Console script: `ml_smell_detector`
- Build backend: hatchling (configured in `pyproject.toml`)
- Supported Python: 3.10+

## Setup

```bash
uv sync --extra dev
```

## Common commands

```bash
# Run the full test suite (212 tests)
uv run python -m pytest tests/

# Lint (both must pass)
uv run ruff check .
uv run python -m flake8 ml_code_smell_detector tests

# Auto-fix lint issues where possible
uv run ruff check . --fix

# Build distributions and validate metadata
uv build
uvx twine check dist/*

# Build documentation
cd docs && uv run sphinx-build -b html source build/html

# Run the tool
uv run ml_smell_detector analyze <path> --output-dir output
```

## Architecture

- `ml_code_smell_detector/cli.py` — `main()` parses args, walks files, calls
  `analyze_file()`, writes `analysis_report.txt` and `analysis_report.csv`.
- `ml_code_smell_detector/detectors/` — three detector classes, each exposing
  `detect_smells(file_path)`, `get_results()`, and `generate_report()`:
  - `FrameworkSpecificSmellDetector` (`framework_detector.py`) — Pandas, NumPy,
    Sklearn, PyTorch, TensorFlow
  - `HuggingFaceSmellDetector` (`huggingface_detector.py`) — HF Transformers
  - `ML_SmellDetector` (`ml_detector.py`) — general ML practices
- `ml_code_smell_detector/utils.py` — `astroid`-based AST helpers. Import node
  types from `astroid.nodes` (e.g. `nodes.Call`), not the top-level `astroid`
  namespace (those aliases are deprecated and removed in astroid v5).

Each detector's `visit_module()` calls individual `detect_*` / `check_*` methods
that append smell dicts to `self.smells`. Every smell dict must contain the keys:
`name`, `framework`, `fix`, `benefits`, `location`.

## Conventions

- Line length: 150 (see `.flake8` and `[tool.ruff]` in `pyproject.toml`).
- Target Python 3.10: do **not** use PEP 701 multi-line f-string expressions
  (newlines inside `{ ... }`); they are a SyntaxError before 3.12.
- Keep both `ruff check` and `flake8` green before committing.
- Add tests under `tests/` for both detection and non-detection cases.

## CI / Release

- `.github/workflows/ci.yml` runs lint, a Python 3.10–3.13 test matrix, and a
  build+twine check on every push/PR to `main`.
- `.github/workflows/publish.yml` publishes to PyPI via Trusted Publishing
  (OIDC) when a GitHub Release is published — no API token required.
- To release: bump `version` in `pyproject.toml`, update
  `docs/source/changelog.rst`, then publish a GitHub Release.
