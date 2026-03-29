# ML Code Smell Detector

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
- [Building Documentation](#building-documentation)
- [Publishing to PyPI](#publishing-to-pypi)
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
git clone https://github.com/KarthikShivasankar/ml_code_smell_detector
cd ml_code_smell_detector
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

```bash
# Run all tests
uv run pytest tests/

# Single test file
uv run pytest tests/test_framework_detector.py

# Single test case
uv run pytest tests/test_sklearn_cv_detector.py::TestCrossValidationDetector::test_all_cv_files

# With coverage report
uv run pytest tests/ --cov=ml_code_smell_detector
```

---

## Building Documentation

```bash
# Windows
rebuild_docs.bat

# Manual
cd docs && sphinx-build -b html source build/html
```

---

## Publishing to PyPI

### Prerequisites

1. Create an account at [pypi.org](https://pypi.org/account/register/)
2. Go to **Account Settings → API tokens** and create a token
3. Store the token — you will only see it once

### Build and publish

```bash
# Build sdist and wheel into dist/
uv build

# Publish (prompts for credentials)
uv publish

# Or pass the token directly
uv publish --token pypi-<your-token-here>
```

### Publish to TestPyPI first (recommended)

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<your-test-token>

# Verify the test install
uv pip install --index-url https://test.pypi.org/simple/ ml-code-smell-detector
```

### Bump the version

Edit `version` in `pyproject.toml`, then build and publish again.

---

## License

MIT
