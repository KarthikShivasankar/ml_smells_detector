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
- [Development & Maintenance](#development--maintenance)
- [Running Tests](#running-tests)
- [Linting](#linting)
- [Documentation](#documentation)
- [Continuous Integration](#continuous-integration)
- [Releasing a New Version](#releasing-a-new-version)
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
ml_smell_detector analyze <path> [<path> ...] [options]
python -m ml_code_smell_detector analyze <path> ...     # same thing
```

| Argument | Description |
|---|---|
| `path ...` | One or more `.py` files or directories; `-` reads source from stdin |
| `--format {text,json,sarif}` | `text` (default) writes the reports below; `json` / `sarif` print one document to stdout and write no files |
| `--output-dir DIR` | Directory to write text reports to (default: `output/`) |
| `--ignore DIR [DIR ...]` | Directory names to skip (`.venv`, `venv`, `.git`, `node_modules`, `__pycache__`, `build`, `dist`, caches and hidden directories are always skipped) |
| `--select RULE [RULE ...]` | Only report rule ids starting with these prefixes, e.g. `ML.PANDAS ML.DATA_LEAKAGE` |
| `--ignore-rules RULE [RULE ...]` | Drop rule ids starting with these prefixes, e.g. `ML.MISSING_DOCSTRING` |
| `--stdin-filename NAME` | File name to analyse stdin as (default `stdin.py`) |
| `--fail-on-findings` | Exit with status 1 if anything is reported (CI gates, agent loops) |
| `--version` | Print the version |

Exit codes: `0` success, `1` findings with `--fail-on-findings`, `2` usage error (e.g. a path that does not exist).

`ml_smell_detector rules` prints every rule id the detectors can emit, with how-to-fix and benefit text, as JSON.

### Silencing findings

Add a comment to the analysed code, by rule id or rule-id prefix:

```python
x = df.values  # mlscent: ignore[ML.PANDAS.DATAFRAME_CONVERSION_CHECKER]
df.dropna(inplace=True)  # mlscent: ignore            <- every rule on this line
# mlscent: ignore-file[ML.SCIKITLEARN.UNIT_TESTING_CHECKER, ML.MISSING_DOCSTRING]
```

File-level findings (reported at line 0) can only be silenced with `ignore-file` or `--ignore-rules`.

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

# Only the files changed on this branch, as JSON
git diff --name-only main -- '*.py' | xargs ml_smell_detector analyze --format json

# Pipe code in
cat train.py | ml_smell_detector analyze - --format json --select ML.DATA_LEAKAGE ML.SCIKITLEARN
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
  uses: actions/upload-artifact@v7
  with:
    name: smell-report
    path: smell_report/
```

Show findings as GitHub code-scanning alerts with SARIF:

```yaml
- run: ml_smell_detector analyze ./src/ --format sarif > mlscent.sarif
- uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: mlscent.sarif
```

Gate a build on specific rules: `ml_smell_detector analyze src/ --format json --select ML.DATA_LEAKAGE --fail-on-findings`.

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

### 6. Use from an agent, harness or script

MLScent is built to be called by other tools: deterministic output, stable rule ids, nothing printed by the
library, meaningful exit codes, and no ML framework needed at runtime (it only parses code).

**Python API** (`ml_code_smell_detector.api`, also importable from the package root):

```python
from ml_code_smell_detector import analyze_file, analyze_paths, analyze_source, list_rules

report = analyze_paths(["src/", "train.py"], select=["ML.PANDAS"], ignore=["ML.PANDAS.DATATYPE_CHECKER"])
for result in report["files"]:
    for f in result["findings"]:                   # sorted by (line, rule_id); line 0 = file-level
        print(result["file"], f["line"], f["rule_id"], f["how_to_fix"])
print(report["summary"])                           # {"files", "files_with_findings", "findings", "suppressed", "errors", "by_rule"}

result = analyze_source(code, filename="train.py") # one file from a string; filename matters (see below)
```

**CLI**: `ml_smell_detector analyze src/ --format json` prints the same report; `--format sarif` prints SARIF 2.1.0.

Report document (`schema_version` is bumped on incompatible changes):

```json
{
  "tool": "ml-code-smell-detector", "version": "0.2.0", "schema_version": 1,
  "files": [{
    "file": "src/train.py",
    "findings": [{
      "rule_id": "ML.DATA_LEAKAGE", "detector": "general", "framework": "General ML",
      "name": "Potential data leakage", "message": "Potential data leakage: Preprocessing applied before train-test split",
      "line": 14, "snippet": "StandardScaler().fit_transform(features)",
      "how_to_fix": "Split the data first, then fit preprocessing on the training split only (or put it in a Pipeline).",
      "benefits": "Keeps test data out of fitted statistics, so evaluation scores are honest."
    }],
    "errors": [], "suppressed": 0
  }],
  "summary": {"files": 1, "files_with_findings": 1, "findings": 1, "suppressed": 0, "errors": 0,
              "by_rule": {"ML.DATA_LEAKAGE": 1}}
}
```

Notes for tool builders:

- Rule ids are `ML.<FRAMEWORK>.<NAME>` for framework rules (`ML.PYTORCH.GRADIENT_CLEAR_CHECKER`), `ML.HF.<NAME>`
  for Hugging Face rules and short ids such as `ML.MAGIC_NUMBER` for general rules. They are stable; filter and
  suppress by them. `list_rules()` / `ml_smell_detector rules` is the catalogue.
- `line` is 1-based; `0` marks a file-level heuristic (e.g. "no logging in a training script") rather than a
  specific statement. Treat line findings as more precise than file-level ones.
- Parse and detector errors are returned in `errors` (one per problem), never raised or printed; a crash in one
  check does not stop the others.
- Some general checks key off the file *name* (not its directory): files named like `test_*.py`, `*utils*` or
  `*data*` skip checks that do not apply to them. Pass a realistic `filename=` to `analyze_source`.
- The detectors are heuristics over the AST (no type inference). Expect some false positives on file-level
  rules; use `--select` to focus on the precise ones, or suppression comments to accept a finding.

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
- Missing random seed, early stopping, checkpointing, logging
- Models built or trained in a loop without `clear_session()` (memory release)
- Unmasked `tf.math.log` of risky inputs
- Python lists grown in a loop inside `@tf.function` (use `tf.TensorArray`)
- Threshold-dependent metrics only (no AUC)
- Missing batch normalisation, dropout, data augmentation, learning-rate scheduling, evaluation
- Missing unit tests; `load_model` / `saved_model` I/O without exception handling

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

## Development & Maintenance

This section is for contributors and maintainers of the package itself.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (package/environment manager)
- Python 3.10+ (CI tests 3.10–3.14)

### Setup

```bash
git clone https://github.com/KarthikShivasankar/ml_smells_detector
cd ml_smells_detector
uv sync --extra dev          # creates .venv and installs the package + dev tools
```

`uv sync --extra dev` installs everything in `[project.optional-dependencies].dev`
(pytest, pytest-cov, ruff, flake8, Sphinx, …). Run any tool with `uv run <cmd>`.

### Project layout

```
ml_code_smell_detector/
  cli.py                 # CLI entry point: arg parsing, file walking, report writing
  utils.py               # astroid-based AST helpers (import node types from astroid.nodes)
  detectors/
    framework_detector.py    # Pandas / NumPy / sklearn / PyTorch / TensorFlow
    huggingface_detector.py  # Hugging Face Transformers
    ml_detector.py           # general ML practices
tests/                   # pytest suite (mirrors the detectors)
docs/source/             # Sphinx documentation sources
.github/workflows/       # CI and release automation
AGENTS.md                # quick command/convention reference for AI agents
```

### The development loop

```bash
uv run ruff check . --fix                       # lint + auto-fix (incl. import order)
uv run python -m flake8 ml_code_smell_detector tests
uv run python -m pytest tests/                  # unit tests, corpus and end-to-end CLI tests
uv build && uvx twine check dist/*              # sanity-check the package
```

Keep **both** `ruff check` and `flake8` green, and all tests passing, before
committing. CI enforces all of this on every push and PR.

### Adding a new smell / detector

1. Add detection logic to the relevant class in `ml_code_smell_detector/detectors/`.
2. Framework rules: add the rule's entry to `get_smells()` and report it with `add_smell(framework, name, node, ...)`.
   Hugging Face rules: add it to `HuggingFaceSmellDetector.RULES`. General rules: add a `(message prefix, rule id,
   how to fix, benefits)` row to `ML_SmellDetector.RULES`. Register the check in the detector's `run_checks(...)` list.
3. Report at the most specific node (the offending call), not the module, whenever the rule is about one statement.
4. Add tests under `tests/` for **both** detection and non-detection cases, and a case in `tests/corpus/`
   (annotated with `# expect: RULE_ID`) — see "Test Structure" below.
5. Document the new smell in `docs/source/features.rst` and the "Detected Smells" list above.
6. Run the development loop and open a PR.

### Coding conventions

- Line length: **150** (configured in `.flake8` and `[tool.ruff]`).
- Target Python 3.10 — do **not** use PEP 701 multi-line f-string expressions
  (newlines inside `{ ... }`); they are a `SyntaxError` before Python 3.12.
- Import `astroid` node types from `astroid.nodes` (e.g. `nodes.Call`), not the
  deprecated top-level `astroid` aliases.

---

## Running Tests

The test suite covers all three detector classes, the API, utilities and the CLI, plus an end-to-end corpus.

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
| `test_api.py` | Unified findings, suppression comments, rule filters, multi-path reports | ~12 |
| `test_corpus.py` | End-to-end corpus: every file in `tests/corpus/` must produce exactly its annotated findings | per file |
| `test_e2e_cli.py` | The CLI run as a subprocess: JSON/SARIF output, exit codes, stdin, rule selection | ~8 |

#### The corpus

`tests/corpus/*.py` are realistic ML scripts annotated with the findings they must produce:

```python
sales.dropna(inplace=True)  # expect: ML.PANDAS.INPLACE_CHECKER
# expect-file: ML.PYTORCH.LOGGING_CHECKER      <- a file-level (line 0) finding
```

`test_corpus.py` fails on any missing finding (recall regression) or extra finding (precision regression).
`clean_pipeline.py` must stay finding-free. After an intentional rule change, review the new output and update
the annotations. The corpus is excluded from ruff/flake8 because it is smelly on purpose.

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

## Documentation

The docs are built with [Sphinx](https://www.sphinx-doc.org/) from reStructuredText
sources in `docs/source/` and are hosted on Read the Docs.

### Where to edit

| File | Contents |
|---|---|
| `docs/source/index.rst` | Landing page / table of contents |
| `docs/source/installation.rst` | Install instructions (keep the Python version in sync with `pyproject.toml`) |
| `docs/source/usage.rst` | CLI usage and options |
| `docs/source/features.rst` | Full list of detected smells |
| `docs/source/detectors/*.rst` | Auto-generated API docs for each detector |
| `docs/source/contributing.rst` | Contributor guide |
| `docs/source/changelog.rst` | Per-version changelog (update on every release) |
| `docs/source/conf.py` | Sphinx config — bump `release` on every version bump |

### Build locally

```bash
# Windows
rebuild_docs.bat

# Any platform
uv run sphinx-build -b html docs/source docs/build/html
# then open docs/build/html/index.html
```

> `docs/build/` is generated output and is **git-ignored** — never commit it.

### Published docs

Read the Docs rebuilds automatically on every push to `main` using
`.readthedocs.yaml` (Ubuntu 24.04, Python 3.13; installs the package so autodoc can import it) and
`docs/requirements.txt`. CI builds the docs the same way with warnings as errors. When you change
the public API or add a detector, update `features.rst` and the relevant
`detectors/*.rst` so the published docs stay accurate.

---

## Continuous Integration

GitHub Actions workflows live in `.github/workflows/`:

- **`ci.yml`** — runs on every push and pull request to `main` (and as the release gate, via `workflow_call`):
  - **Lint**: `uv lock --check` (lockfile in sync with `pyproject.toml`), `ruff check`, `flake8`
  - **Test**: full suite (unit, corpus, CLI end-to-end) across Python 3.10–3.14, with coverage
    (`fail_under = 90` in `pyproject.toml`)
  - **Oldest deps**: the suite on Python 3.10 with the declared minimums (astroid 3.0.0, tqdm 4.60.0)
  - **Docs**: Sphinx build as on Read the Docs, warnings are errors
  - **Build**: `uv build`, `twine check --strict`, then the wheel is installed into a clean virtualenv and run
    from outside the checkout (CLI, `python -m`, JSON and SARIF over the corpus), so a module or entry point missing
    from the wheel fails CI. The checked `dist/` is uploaded as an artifact.
- **`publish.yml`** — the release pipeline (see below).
- **`.github/dependabot.yml`** — weekly grouped updates for GitHub Actions and `uv.lock`, monthly for
  `docs/requirements.txt`.

Workflow runs are visible under the repo's **Actions** tab. A red CI run blocks a release.

---

## Releasing a New Version

Releases use [semantic versioning](https://semver.org/). Publishing a GitHub Release runs `publish.yml`:

1. **verify** — the tag (`vX.Y.Z`) must equal `version` in `pyproject.toml`, and `docs/source/changelog.rst` must
   have an `X.Y.Z (...)` section;
2. **ci** — the whole CI workflow above runs on the tag;
3. **pypi** — the exact distributions CI built and smoke-tested are uploaded to PyPI with Trusted Publishing
   (PyPI attestations are generated automatically);
4. **github-release** — the same files are attached to the GitHub Release.

Checklist:

1. Make sure `main` is green.
2. Bump `version` in `pyproject.toml` (the docs read it from there) and rename the changelog's **Unreleased**
   section to `X.Y.Z (YYYY-MM-DD)`. `tests/test_release.py` checks these agree.
3. Commit and push to `main`, and wait for CI.
4. Create the release from the changelog section:
   ```bash
   VERSION=X.Y.Z
   awk -v v="$VERSION" '$0 ~ "^"v" \\(" {f=1; getline; next} f && /^[0-9]+\.[0-9]+\.[0-9]+ \(/ {exit} f' \
       docs/source/changelog.rst > /tmp/notes.rst
   gh release create "v$VERSION" --target main --title "v$VERSION" --notes-file /tmp/notes.rst
   ```
5. Watch **Actions → Publish to PyPI**, then verify the release installs:
   ```bash
   uvx --from "ml-code-smell-detector==$VERSION" ml_smell_detector --version
   ```

If publishing fails after the release exists (for example a missing Trusted Publisher), fix the cause and re-run
**Actions → Publish to PyPI → Run workflow** on the tag; uploads use `skip-existing`, so re-runs are safe.

> If a release is broken (e.g. fails to import on a supported Python version),
> **yank** it on PyPI (Manage → Release → Yank) and ship a fixed patch release.
> Yanking hides it from new installs without deleting it.

---

## Publishing to PyPI

### Trusted Publishing (OIDC, no token)

`publish.yml` uploads to PyPI using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API
token is stored or pasted anywhere. It requires a **one-time setup** on the PyPI project page
(**Your projects → ml-code-smell-detector → Settings → Publishing → Add a new publisher → GitHub**):

| Field | Value |
|---|---|
| Owner | `KarthikShivasankar` |
| Repository | `ml_smells_detector` |
| Workflow name | `publish.yml` |
| Environment | `pypi` |

Without it the upload fails with `invalid-publisher` (valid token, but no corresponding publisher).

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
