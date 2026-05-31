# ML Code Smell Detector

[![CI](https://github.com/KarthikShivasankar/ml_smells_detector/actions/workflows/ci.yml/badge.svg)](https://github.com/KarthikShivasankar/ml_smells_detector/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/ml-code-smell-detector.svg)](https://pypi.org/project/ml-code-smell-detector/)
[![Python versions](https://img.shields.io/pypi/pyversions/ml-code-smell-detector.svg)](https://pypi.org/project/ml-code-smell-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A static analysis CLI tool that detects code smells in Python ML projects — without requiring any ML frameworks to be installed. It uses AST-based analysis (via `astroid`) to identify bad practices across Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, and Hugging Face Transformers.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Use Cases](#use-cases)
- [Output](#output)
- [Detection Scope](#detection-scope)
- [Detected Smells](#detected-smells)
- [Running Tests](#running-tests)
- [Linting](#linting)
- [Building Documentation](#building-documentation)
- [Continuous Integration](#continuous-integration)
- [Publishing to PyPI](#publishing-to-pypi)
- [Citation](#citation)
- [License](#license)

---

## Installation

### From PyPI

Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install the package:

```bash
uv pip install ml-code-smell-detector

# or with pip
pip install ml-code-smell-detector
```

### Development Install

```bash
git clone https://github.com/KarthikShivasankar/ml_smells_detector
cd ml_smells_detector
uv pip install -e ".[dev]"
```

---

## Quick Start

```bash
# Analyze a single file
ml_smell_detector analyze my_model.py

# Analyze an entire project directory
ml_smell_detector analyze ./my_ml_project/

# Save results to a custom folder
ml_smell_detector analyze ./my_ml_project/ --output-dir reports/
```

Reports are written to `analysis_report.txt` and `analysis_report.csv` in the output directory.

---

## Usage

```
ml_smell_detector analyze <path> [options]
```

| Argument | Description |
|---|---|
| `path` | Path to a `.py` file or a directory |
| `--output-dir DIR` | Directory to write reports to (default: `output/`) |
| `--ignore DIR [DIR ...]` | Directory names to skip during analysis |

### Examples

```bash
# Analyze a single training script
ml_smell_detector analyze train.py

# Analyze a full project, save to a custom output dir
ml_smell_detector analyze ./src/ --output-dir ./analysis_results/

# Analyze a project but skip test and notebook folders
ml_smell_detector analyze ./project/ --ignore tests notebooks __pycache__

# Analyze a Jupyter notebook export
ml_smell_detector analyze ./exported_notebook.py --output-dir ./nb_report/
```

---

## Use Cases

### 1. Pre-commit / PR review check

Catch smells before merging ML code changes:

```bash
ml_smell_detector analyze ./ml_code_smell_detector/ --output-dir ./lint_output/ --ignore __pycache__
cat lint_output/analysis_report.txt
```

### 2. Audit an existing ML project

Get a full picture of technical debt in a research or production codebase:

```bash
ml_smell_detector analyze ./research_project/ --output-dir ./audit/ --ignore .git __pycache__ data
```

Then open `audit/analysis_report.csv` in Excel or any spreadsheet tool — each row is a smell with its location, fix, and benefits.

### 3. Compare model training scripts

Analyze multiple scripts and diff the CSV outputs to track quality improvements over iterations:

```bash
ml_smell_detector analyze ./v1/train.py --output-dir ./reports/v1/
ml_smell_detector analyze ./v2/train.py --output-dir ./reports/v2/
```

### 4. Integrate into CI/CD

Add to a GitHub Actions workflow (no ML dependencies needed on the runner):

```yaml
- name: Run ML smell detector
  run: |
    pip install ml-code-smell-detector
    ml_smell_detector analyze ./src/ --output-dir ./smell_report/ --ignore tests
- name: Upload smell report
  uses: actions/upload-artifact@v3
  with:
    name: smell-report
    path: smell_report/
```

### 5. Use as a Python library

```python
from ml_code_smell_detector import (
    FrameworkSpecificSmellDetector,
    HuggingFaceSmellDetector,
    ML_SmellDetector,
)

# Run all detectors on a file
for DetectorClass in [FrameworkSpecificSmellDetector, HuggingFaceSmellDetector, ML_SmellDetector]:
    detector = DetectorClass()
    detector.detect_smells("train.py")
    for smell in detector.get_results():
        print(f"[{smell['framework']}] {smell['name']} @ {smell['location']}")
        print(f"  Fix: {smell['fix']}")
```

---

## Output

Each run produces two report files in the output directory:

### `analysis_report.txt`

Human-readable summary grouped by file and detector category:

```
Analysis results for train.py:

Framework-Specific Smells:
- Missing Random Seed (NumPy)
  Framework: NumPy
  How to fix: Add np.random.seed() at the start of your script
  Benefits: Reproducible experiments
  Location: Line 12

Smell Counts:
  Missing Random Seed: 1
Total smells detected: 1
```

### `analysis_report.csv`

Machine-readable table with columns:

| Framework | Smell/Checker Name | How to Fix | Benefits | File Path | Location | Count |
|---|---|---|---|---|---|---|
| NumPy | Missing Random Seed | Add np.random.seed()... | Reproducible... | train.py | Line 12 | 1 |

Useful for filtering, sorting, or tracking smell trends over time in a spreadsheet or BI tool.

---

## Detection Scope

The tool analyzes all Python code in a file regardless of nesting depth — module-level code, class bodies, class methods, nested functions, and closures.

**Import detection** uses prefix matching, so all of the following are recognized:

```python
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
```

The same applies to `pandas`, `numpy`, `torch`, `tensorflow`, and `transformers`.

---

## Detected Smells

### Framework-Specific Smells (`FrameworkSpecificSmellDetector`)

**Pandas**
- Unnecessary iteration (`iterrows`)
- Chain indexing
- Inefficient merge operations
- Inplace operations
- Inefficient DataFrame conversion (`.values` vs `.to_numpy()`)
- Missing data type specifications
- Column selection issues
- DataFrame mutation during iteration

**NumPy**
- NaN equality checks (use `np.isnan()`)
- Missing random seed
- Inefficient array creation (missing `dtype`)
- Suboptimal element-wise operations
- Dtype inconsistency
- Implicit broadcasting risks
- Copy/view confusion
- Missing axis specification

**Scikit-learn**
- Missing feature scaling
- Absence of Pipeline
- Missing cross-validation
- Inconsistent `random_state`
- Missing verbose mode
- Overreliance on accuracy metric
- Missing unit tests
- Data leakage
- Missing exception handling

**PyTorch**
- Missing `torch.manual_seed()`
- Non-deterministic algorithms
- DataLoader reproducibility
- Missing mask in log operations
- Direct `model.forward()` calls
- Missing gradient zeroing
- Missing batch normalization
- Missing dropout
- Missing data augmentation
- Missing learning rate scheduler
- Missing logging/monitoring
- Missing eval mode

**TensorFlow**
- Missing random seed, early stopping, checkpointing, memory management, logging

---

### Hugging Face Smells (`HuggingFaceSmellDetector`)

- Model versioning issues
- Missing tokenizer and model caching
- Inconsistent tokenization settings
- Inefficient data loading
- Missing distributed training configuration
- Missing mixed precision training
- Missing gradient accumulation
- Missing learning rate scheduling
- Missing early stopping

---

### General ML Smells (`ML_SmellDetector`)

- Data leakage detection
- Magic number usage
- Inconsistent feature scaling
- Missing cross-validation
- Imbalanced dataset handling
- Feature selection issues
- Overreliance on single metrics
- Missing model persistence
- Missing reproducibility measures
- Inefficient data loading for large datasets
- Unused feature detection
- Overfitting-prone practices
- Missing error handling
- Hardcoded file paths
- Missing or incomplete documentation

---

## Running Tests

The test suite has **212 tests** covering all three detector classes, utilities, and the CLI.

```bash
# Run the full test suite
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_pandas_smells.py
python -m pytest tests/test_pytorch_smells.py
python -m pytest tests/test_tensorflow_smells.py
python -m pytest tests/test_sklearn_smells.py
python -m pytest tests/test_numpy_smells.py
python -m pytest tests/test_huggingface_smells.py
python -m pytest tests/test_ml_detector.py
python -m pytest tests/test_utils.py
python -m pytest tests/test_cli.py

# Run a single test class or function
python -m pytest tests/test_sklearn_smells.py::TestCrossValidationChecker
python -m pytest tests/test_pytorch_smells.py::TestGradientClearChecker::test_detects_missing_zero_grad

# With coverage report
python -m pytest tests/ --cov=ml_code_smell_detector --cov-report=term-missing
```

### Test Structure

| File | Covers | Tests |
|---|---|---|
| `test_pandas_smells.py` | Pandas smells (Unnecessary Iteration, Chain Indexing, Merge Params, InPlace, etc.) | ~20 |
| `test_numpy_smells.py` | NumPy smells (NaN equality, randomness, axis, dtype, etc.) | ~16 |
| `test_sklearn_smells.py` | Sklearn smells (Scaler, Pipeline, CV, Randomness, Verbose, Threshold, etc.) | ~20 |
| `test_pytorch_smells.py` | PyTorch smells (Randomness, Determinism, Gradients, BatchNorm, Dropout, etc.) | ~20 |
| `test_tensorflow_smells.py` | TensorFlow smells (Randomness, EarlyStopping, Checkpointing, Memory, etc.) | ~20 |
| `test_huggingface_smells.py` | HuggingFace smells (versioning, caching, mixed precision, etc.) | ~18 |
| `test_ml_detector.py` | General ML smells (leakage, magic numbers, CV, reproducibility, etc.) | ~22 |
| `test_utils.py` | AST utility functions | ~30 |
| `test_cli.py` | CLI argument parsing, file collection, report writing | ~10 |

---

## Linting

The project uses [Ruff](https://docs.astral.sh/ruff/) as the primary linter (and import sorter) and keeps `flake8` available as a secondary check. Both are configured for a 150-character line length (`pyproject.toml` `[tool.ruff]` and `.flake8`).

```bash
# Lint with Ruff
uv run ruff check .

# Auto-fix what Ruff can (import order, simple issues)
uv run ruff check . --fix

# Run flake8 as well
uv run python -m flake8 ml_code_smell_detector tests
```

Both linters must pass cleanly before committing. CI runs them on every push and pull request.

---

## Building Documentation

```bash
# Windows
rebuild_docs.bat

# Manual
cd docs && sphinx-build -b html source build/html
```

---

## Continuous Integration

GitHub Actions workflows live in `.github/workflows/`:

- **`ci.yml`** — runs on every push and pull request to `main`:
  - **Lint**: `ruff check` + `flake8`
  - **Test**: full suite across Python 3.10, 3.11, 3.12, and 3.13
  - **Build**: `uv build` + `twine check` to validate the distribution
- **`publish.yml`** — publishes to PyPI when a GitHub Release is published (see below).

---

## Publishing to PyPI

### Recommended: Trusted Publishing (OIDC, no token)

This repo ships a `publish.yml` workflow that uploads to PyPI using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API token is
stored or pasted anywhere.

**One-time setup** on the PyPI project page
(**Settings → Publishing → Add a trusted publisher**):

| Field | Value |
|---|---|
| Owner | `KarthikShivasankar` |
| Repository | `ml_smells_detector` |
| Workflow name | `publish.yml` |
| Environment | `pypi` |

**To release a new version:**

1. Bump `version` in `pyproject.toml` and update `docs/source/changelog.rst`.
2. Commit and push to `main`.
3. Create a GitHub Release (e.g. tag `v0.1.2`). The `publish.yml` workflow builds,
   runs `twine check`, and publishes automatically.

### Manual publish (fallback)

```bash
# Build sdist and wheel into dist/
uv build

# Validate, then publish (prompts for credentials)
uvx twine check dist/*
uv publish

# Or pass a project-scoped token directly
uv publish --token pypi-<your-token-here>
```

> Prefer a **project-scoped** API token over an account-wide one, and never commit
> tokens to the repo. Trusted Publishing avoids tokens entirely.

### Publish to TestPyPI first (optional)

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<your-test-token>

# Verify the test install
uv pip install --index-url https://test.pypi.org/simple/ ml-code-smell-detector
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@inproceedings{shivashankar2025mlscent,
  title     = {MLScent: A tool for Anti-pattern detection in ML projects},
  author    = {Shivashankar, Karthik and Martini, Antonio},
  booktitle = {2025 IEEE/ACM 4th International Conference on AI Engineering--Software Engineering for AI (CAIN)},
  pages     = {150--160},
  year      = {2025},
  month     = {April},
  publisher = {IEEE}
}
```

Shivashankar, K., & Martini, A. (2025, April). MLScent: A tool for Anti-pattern detection in ML projects. In *2025 IEEE/ACM 4th International Conference on AI Engineering–Software Engineering for AI (CAIN)* (pp. 150–160). IEEE.

---

## License

MIT
