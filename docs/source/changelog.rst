Changelog
=========

Unreleased
----------

Agent and harness support

- ``--format sarif`` (SARIF 2.1.0), ``--fail-on-findings`` (exit 1), exit code 2 for usage errors, ``--version``,
  several paths per run, ``-`` for stdin (``--stdin-filename``) and ``python -m ml_code_smell_detector``.
- Rule filters ``--select`` / ``--ignore-rules`` (rule-id prefixes) and in-source suppressions
  ``# mlscent: ignore[RULE]`` / ``# mlscent: ignore-file[RULE]``.
- JSON reports carry ``tool``, ``version``, ``schema_version`` and a ``summary``; ``api.analyze_paths``,
  ``api.to_sarif`` and the API functions are exported from the package root.
- Every rule in ``ml_smell_detector rules`` now has how-to-fix and benefit text (Hugging Face and general rules
  had none), and general findings carry it too.
- Directory scans skip virtualenvs, ``.git``, ``node_modules``, caches and hidden directories, and are sorted.
- Text reports are built from the same findings as JSON, so filters and suppressions apply to them as well.
- Errors are reported once per problem; a crash in one check no longer skips the remaining checks.
- Source files are read with their declared encoding (UTF-8 by default) instead of the platform default.
- New end-to-end tests: an annotated corpus (``tests/corpus``) that pins exact findings, and CLI subprocess tests.

Performance

- About 3x faster on real code (400 ML files: 23.7 s to 8.5 s) with identical findings: each file is parsed once
  and shared by the three detectors, and node rendering / tree walks that every check repeated are memoised
  (profiling showed ``as_string`` and ``nodes_of_class`` dominating).

Dependencies, CI and docs

- ``astroid>=3.0`` (was ``>=2.0``, which never worked: the detectors use ``nodes.Try``, added in astroid 3.0) and
  ``tqdm>=4.60``. Locked to astroid 4.3.4, tqdm 4.70.1, pytest 9.1, ruff 0.17, flake8 7.4 (pyflakes 4); Python 3.14
  is supported and tested.
- CI: actions/checkout v7, astral-sh/setup-uv v10.3.0 (caching via its secure ``auto`` default),
  upload-artifact v7 / download-artifact v8, a Python 3.14 job, a job on the oldest supported dependencies, and a
  docs build with warnings as errors.
- Read the Docs builds on Ubuntu 24.04 / Python 3.13 with current Sphinx and installs the package, so the API pages
  are no longer empty there.

Detection fixes

- Hugging Face: the ``TrainingArguments`` checks (mixed precision, gradient accumulation, distributed training)
  never fired on real code (they matched a variable *named* ``TrainingArguments``); a non-literal argument such as
  ``fp16=USE_AMP`` crashed the detector. Findings are now reported at the ``TrainingArguments(...)`` call.
- Hugging Face: model versioning covers every model class (``BertForMaskedLM``, ``GPT2LMHeadModel``, ...) and
  ``pipeline(model=...)``, and skips local checkpoints; the LR-scheduler rule targets custom training loops
  (``Trainer`` always schedules); data-loading advice is limited to training code; ``import datasets`` is recognised.
- General: checks keyed off the full path, so any file under a directory such as ``data/`` or ``tests/`` silently
  skipped them; they now look at the file name. Plain ``import torch`` / ``import tensorflow as tf`` imports are
  recognised (only ``from ... import`` was).
- General: keyword arguments were invisible to several checks (``class_weight=``, ``stratify=``, version
  arguments, ``chunksize=``) because they were read from node reprs.
- General: data leakage is found even when a model is fitted after the split, and is reported at the offending
  ``fit``/``fit_transform``; magic numbers are literals in comparisons/arithmetic, not named assignments; the
  unused-features check (dead code before) works; ``if __name__ == "__main__"`` no longer counts as validation;
  error handling looks at I/O only; incomplete seeding names the unseeded framework (aliases such as ``tf`` were
  missed); hardcoded paths skip MIME types, ratios and fragments joined to a base.
- Pandas: ``df.merge(b)`` is checked (only ``pd.merge`` was); every ``inplace=True`` is reported; chain indexing
  needs a column label (no more ``grid[i][j] = v``); column selection looks at file reads; dataframe modification
  targets writes inside ``iterrows``/``itertuples`` loops; dtype checks no longer trigger on ``os.path.join``.
- NumPy: NaN comparisons on either side (and ``math.nan``, ``float('nan')``), not ``np.nanmean``; unseeded
  ``default_rng()``; all ``np.random`` draws; ``np.append``/``vstack``/``hstack`` and ``while`` loops for array
  growth; exact reduction names for missing ``axis`` (no ``np.maximum``/``np.cumsum`` false positives); the
  copy-view rule (dead before) works.
- scikit-learn: ``SVC``, k-NN, MLPs and more count as scale-sensitive, and all scalers as scaling; ``make_pipeline``
  is a pipeline; more CV utilities and ``random_state`` APIs are recognised.
- PyTorch: ``DataLoader`` via ``import torch``; ``torch.log`` exactly (not ``log_softmax``/``logsumexp``); any
  ``x.forward()`` except ``super().forward()``; optimizers imported by name; ``range(1, N)`` and named epoch
  counts; LayerNorm/GroupNorm count as normalisation; lowercase ``mlflow``/``neptune`` and ``logging`` count as
  logging; ``model.train(False)`` counts as eval mode; augmentation advice needs a vision signal.
- Found by running on pytorch/examples, scikit-learn, keras-io, nanoGPT and Hugging Face examples: ``forward`` on
  RPC proxies and a module's own ``self.forward`` are not reported, a loop that stacks Keras blocks is not a
  memory leak, and ``a = a[1:]`` is not copy-view confusion.
- TensorFlow: ``fit()`` on any variable (only ``model.fit``); named epoch counts; memory release targets models
  built in a loop; tensor-array advice targets lists grown inside ``@tf.function``; ``Unit Testing Checker`` and
  ``Exception Handling Checker`` (listed but never emitted) are implemented; ``metrics=['accuracy']`` is seen.

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
