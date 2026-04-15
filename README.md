# ML Code Smell Detector

ML Code Smell Detector is a Python package that statically analyzes Python ML code to identify potential issues and bad practices. It includes detectors for framework-specific smells (Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow), Hugging Face Transformers, and general ML patterns.

## Installation

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
```

### Development install

```bash
git clone https://github.com/KarthikShivasankar/ml_code_smell_detector
cd ml_code_smell_detector
uv pip install -e ".[dev]"
```

## Usage

```bash
# Analyze a single file
ml_smell_detector analyze path/to/file.py

# Analyze a directory
ml_smell_detector analyze path/to/project/

# Specify output directory (default: ./output)
ml_smell_detector analyze path/to/project/ --output-dir results/

# Ignore specific folders during directory analysis
ml_smell_detector analyze path/to/project/ --ignore tests docs __pycache__
```

Reports are written to `analysis_report.txt` and `analysis_report.csv` in the output directory. Progress is displayed via a progress bar when analyzing directories.

## Running Tests

```bash
uv run pytest tests/

# Single test file
uv run pytest tests/test_framework_detector.py

# With coverage
uv run pytest tests/ --cov=ml_code_smell_detector
```

## Publishing to PyPI

### Prerequisites

1. Create an account at [pypi.org](https://pypi.org/account/register/)
2. Go to **Account Settings → API tokens** and create a token scoped to the project (or `--scope=project` for a new upload)
3. Store the token — you will only see it once

### Build and publish

```bash
# Build the sdist and wheel into dist/
uv build

# Publish to PyPI — uv will prompt for credentials
uv publish

# Or pass the token directly
uv publish --token pypi-<your-token-here>
```

### Publish to TestPyPI first (recommended for first release)

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token pypi-<your-test-token>
```

Verify the test install:

```bash
uv pip install --index-url https://test.pypi.org/simple/ ml-code-smell-detector
```

### Bump the version

Edit `version` in `pyproject.toml`, then build and publish again.

## Detection Scope

The tool analyzes all Python code in a file regardless of nesting depth. Smells are detected inside:

- Module-level code
- Class bodies and class methods
- Nested functions and closures

**Import detection** uses prefix matching, so all of the following are recognized:

```python
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
```

The same applies to `pandas`, `numpy`, `torch`, `tensorflow`, and `transformers`.

## Features

### Framework-Specific Smell Detector

**Pandas:**
- Unnecessary iteration (iterrows)
- Chain indexing
- Inefficient merge operations
- Inplace operations
- Inefficient DataFrame conversion (`.values` vs `.to_numpy()`)
- Missing data type specifications
- Column selection issues
- DataFrame mutation during iteration

**NumPy:**
- NaN equality checks (use `np.isnan()`)
- Missing random seed
- Inefficient array creation (missing `dtype`)
- Suboptimal element-wise operations
- Dtype inconsistency
- Implicit broadcasting risks
- Copy/view confusion
- Missing axis specification

**Scikit-learn:**
- Missing feature scaling
- Absence of Pipeline
- Missing cross-validation
- Inconsistent `random_state`
- Missing verbose mode
- Overreliance on accuracy metric
- Missing unit tests
- Data leakage
- Missing exception handling

**PyTorch:**
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

**TensorFlow:**
- Missing random seed, early stopping, checkpointing, memory management, logging

### Hugging Face Smell Detector

- Model versioning issues
- Missing tokenizer and model caching
- Inconsistent tokenization settings
- Inefficient data loading
- Missing distributed training configuration
- Missing mixed precision training
- Missing gradient accumulation
- Missing learning rate scheduling
- Missing early stopping

### General ML Smell Detector

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

## Building Documentation

```bash
cd docs && sphinx-build -b html source build/html
```

## License

MIT
