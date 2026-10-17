# AGENTS.md

Guidance for AI coding agents working in this repository, and for agents or harnesses that use MLScent as a tool.

## Project

`ml-code-smell-detector` is a Python CLI and library that statically analyzes Python ML code using AST (via
`astroid`) to detect code smells across Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow, and Hugging Face
Transformers. It performs static analysis only and does **not** require any ML framework to be installed.

- Package: `ml_code_smell_detector/`
- Distribution name (PyPI): `ml-code-smell-detector`
- Console script: `ml_smell_detector` (also `python -m ml_code_smell_detector`)
- Build backend: hatchling (configured in `pyproject.toml`)
- Supported Python: 3.10–3.14; runtime deps `astroid>=3.0`, `tqdm>=4.60`

## Setup

```bash
uv sync --extra dev
```

## Common commands

```bash
# Full test suite: unit tests, the annotated corpus (tests/corpus) and CLI subprocess tests
uv run python -m pytest tests/

# Lint (both must pass)
uv run ruff check .
uv run python -m flake8 ml_code_smell_detector tests

# Auto-fix lint issues where possible
uv run ruff check . --fix

# Build distributions and validate metadata
uv build
uvx twine check dist/*

# Build documentation (CI builds with -W: warnings are errors)
uv run sphinx-build -W -b html docs/source docs/build/html

# Run the tool
uv run ml_smell_detector analyze <path> --output-dir output
```

## Using MLScent as a tool from an agent or harness

```bash
ml_smell_detector analyze src/ train.py --format json        # one JSON report on stdout, no files written
ml_smell_detector analyze src/ --format sarif                # SARIF 2.1.0 (code scanning, IDEs)
cat train.py | ml_smell_detector analyze - --format json     # stdin (--stdin-filename to name it)
ml_smell_detector analyze src/ --format json --select ML.DATA_LEAKAGE ML.PANDAS --ignore-rules ML.PANDAS.DATATYPE_CHECKER
ml_smell_detector analyze src/ --format json --fail-on-findings   # exit 1 if anything is reported
ml_smell_detector rules                                     # rule catalogue with how-to-fix text
```

Exit codes: `0` ok, `1` findings with `--fail-on-findings`, `2` usage error (missing path). Nothing but the document
goes to stdout in json/sarif mode.

From Python, prefer `ml_code_smell_detector.api` (also re-exported from the package root) over the detector
classes: `analyze_paths(paths, exclude_dirs=(), select=None, ignore=None)` returns
`{"tool", "version", "schema_version", "files": [...], "summary": {...}}`; `analyze_file(path)` /
`analyze_source(code, filename=...)` return `{"file", "findings", "errors", "suppressed"}`. A finding is
`rule_id`, `detector`, `framework`, `name`, `message`, `line` (1-based; 0 = file-level heuristic), `snippet`,
`how_to_fix`, `benefits`; findings are sorted by `(line, rule_id)`. Nothing is printed; detector errors are returned
in `errors`. `list_rules()` is the rule catalogue; `to_sarif(report)` converts a report.

Silence findings in analysed code with `# mlscent: ignore[RULE_ID_PREFIX, ...]` on the line, or
`# mlscent: ignore-file[...]` anywhere in the file (bare `ignore` / `ignore-file` silence everything).

Rule ids are stable: keep them unchanged when editing a rule (a framework rule's id is derived from its name, so
renaming a smell changes its id). Bump `api.SCHEMA_VERSION` when the shape of a finding or report changes.

## Architecture

- `ml_code_smell_detector/api.py` — machine-friendly layer: runs the three detectors, unifies findings, rule ids,
  filters, suppressions, reports, SARIF. The CLI is built on it.
- `ml_code_smell_detector/cli.py` — `main(argv)` parses args and prints JSON/SARIF or writes
  `analysis_report.txt` / `analysis_report.csv` (text format) from `api` results.
- `ml_code_smell_detector/detectors/` — three detector classes, each exposing `detect_smells(file_path)`,
  `get_results()`, and `generate_report()`:
  - `FrameworkSpecificSmellDetector` (`framework_detector.py`) — Pandas, NumPy, Sklearn, PyTorch, TensorFlow;
    rule texts in `get_smells()`
  - `HuggingFaceSmellDetector` (`huggingface_detector.py`) — HF Transformers; rule texts in `RULES`
  - `ML_SmellDetector` (`ml_detector.py`) — general ML practices; `(message prefix, rule id, fix, benefits)` in `RULES`
- `ml_code_smell_detector/utils.py` — `astroid` helpers: `parse_file` (one cached parse per file content, shared by
  the detectors), memoised `source(node)` / `nodes_of(node, cls)` (use these instead of `node.as_string()` /
  `node.nodes_of_class()` in checks: they dominated the profile), `get_import_map` / `resolve_call_name`
  (resolve `np.zeros` to `numpy.zeros`), `callee_name`, `keyword`, `int_value`, `in_loop`, `run_checks`.
  Import node types from `astroid.nodes` (e.g. `nodes.Call`), not the top-level `astroid` namespace.

Each detector's `visit_module()` passes its checks to `run_checks(...)`, which isolates them: a crash in one check is
reported on stderr (captured into `errors` by the API) and the others still run. Checks must treat the AST as
read-only (parses and node strings are cached).

## Conventions

- Line length: 150 (see `.flake8` and `[tool.ruff]` in `pyproject.toml`).
- Target Python 3.10: do **not** use PEP 701 multi-line f-string expressions (newlines inside `{ ... }`); they are a
  SyntaxError before 3.12.
- Match on resolved names (`resolve_call_name`, `callee_name`) rather than substrings of `as_string()`: substring
  matching is the main source of false positives (`'log' in 'log_softmax'`, `'Pipeline' not in 'make_pipeline'`).
- Report at the offending node (call, assignment) when a rule is about one statement; use the module (line 0) only
  for file-level heuristics.
- Keep both `ruff check` and `flake8` green before committing.
- Add tests under `tests/` for both detection and non-detection cases, and extend `tests/corpus/`: each corpus file
  is annotated `# expect: RULE_ID` per line and `# expect-file: RULE_ID` for line-0 findings, and
  `tests/test_corpus.py` requires an exact match (missing = recall regression, extra = precision regression).
  `clean_pipeline.py` must stay finding-free. The corpus is excluded from linting on purpose.
- Before claiming a detection change is safe, run it over real code (e.g. shallow clones of pytorch/examples,
  scikit-learn/examples, keras-io/examples) and diff the findings against the previous version; read the code behind
  a sample of findings rather than trusting counts or exit codes.

## CI / Release

- `.github/workflows/ci.yml` runs lint, a Python 3.10–3.14 test matrix, a job against the oldest supported
  dependencies (Python 3.10, astroid 3.0.0, tqdm 4.60.0), a docs build with warnings as errors (as on Read the
  Docs), and a build + twine check on every push/PR to `main`.
- `.github/workflows/publish.yml` publishes to PyPI via Trusted Publishing (OIDC) when a GitHub Release is
  published — no API token required.
- `astral-sh/setup-uv` publishes no moving major tag: pin a full release (e.g. `@v10.3.0`).
- To release: bump `version` in `pyproject.toml` (and `release` in `docs/source/conf.py`), move the changelog's
  "Unreleased" section under the new version in `docs/source/changelog.rst`, then publish a GitHub Release.
