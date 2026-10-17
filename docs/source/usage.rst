Usage
=====

Command-line Interface
----------------------

Analyze a single Python file:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/file.py

Analyze all Python files in a directory:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/directory

Save reports to a custom output directory:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/directory --output-dir reports/

Skip specific directories during analysis:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/directory --ignore tests notebooks __pycache__

Several paths at once, or source from stdin:

.. code-block:: bash

   ml_smell_detector analyze train.py src/ notebooks/export.py
   cat train.py | ml_smell_detector analyze - --format json

Options
-------

- ``path ...`` — one or more ``.py`` files or directories; ``-`` reads source from stdin.
- ``--format {text,json,sarif}`` — ``text`` (default) writes the reports below; ``json`` and ``sarif`` print one
  document to stdout and write no files.
- ``--output-dir DIR`` — directory to write text reports to (default: ``output/``).
- ``--ignore DIR [DIR ...]`` — directory names to skip. Virtualenvs, ``.git``, ``node_modules``, ``__pycache__``,
  ``build``, ``dist``, caches and hidden directories are always skipped.
- ``--select RULE [RULE ...]`` / ``--ignore-rules RULE [RULE ...]`` — keep / drop rule ids starting with these prefixes.
- ``--stdin-filename NAME`` — the file name stdin is analysed as (default ``stdin.py``).
- ``--fail-on-findings`` — exit with status 1 if anything is reported.
- ``--version`` — print the version.

Exit codes: ``0`` success, ``1`` findings with ``--fail-on-findings``, ``2`` usage error (e.g. a missing path).
``python -m ml_code_smell_detector`` is equivalent to ``ml_smell_detector``.

Silencing findings
------------------

Add a comment to the analysed code, by rule id or rule-id prefix:

.. code-block:: python

   x = df.values  # mlscent: ignore[ML.PANDAS.DATAFRAME_CONVERSION_CHECKER]
   df.dropna(inplace=True)  # mlscent: ignore  (every rule on this line)
   # mlscent: ignore-file[ML.SCIKITLEARN.UNIT_TESTING_CHECKER, ML.MISSING_DOCSTRING]

File-level findings (line 0) can only be silenced with ``ignore-file`` or ``--ignore-rules``.

Machine-readable output and the Python API
------------------------------------------

``--format json`` prints ``{"tool", "version", "schema_version", "files": [...], "summary": {...}}``. Each file has
``findings`` (``rule_id``, ``detector``, ``framework``, ``name``, ``message``, ``line``, ``snippet``,
``how_to_fix``, ``benefits``; sorted by line, ``line`` 0 = file-level), ``errors`` and a ``suppressed`` count.
``--format sarif`` prints SARIF 2.1.0 for GitHub code scanning and IDEs. ``ml_smell_detector rules`` prints the
rule catalogue.

The same is available from Python, without printing anything:

.. code-block:: python

   from ml_code_smell_detector import analyze_file, analyze_paths, analyze_source, list_rules

   report = analyze_paths(["src/"], select=["ML.PANDAS"])
   result = analyze_source(code, filename="train.py")

Output
------

The tool writes two report files to the output directory (default: ``output/``):

- ``analysis_report.txt`` — human-readable report grouped by file and detector category
- ``analysis_report.csv`` — machine-readable CSV with columns: ``Framework``, ``Smell/Checker Name``, ``How to Fix``, ``Benefits``, ``File Path``, ``Location``, ``Count``

Use Cases
---------

1. **Code Review**: Automatically check for common ML code smells during code reviews.
2. **Continuous Integration**: Integrate into your CI/CD pipeline to catch potential issues early.
3. **Education**: Teach best practices in ML code to students or junior developers.
4. **Refactoring**: Identify areas of improvement in existing ML codebases.
5. **Project Onboarding**: Quickly assess the quality of ML code in a new project.

CI/CD Integration
-----------------

Add to your GitHub Actions workflow:

.. code-block:: yaml

   - name: Run ML smell detector
     run: |
       pip install ml-code-smell-detector
       ml_smell_detector analyze src/ --output-dir reports/
   - name: Upload reports
     uses: actions/upload-artifact@v7
     with:
       name: ml-smell-reports
       path: reports/

Or report findings as code-scanning alerts, and fail the job on the rules you care about:

.. code-block:: yaml

   - run: ml_smell_detector analyze src/ --format sarif > mlscent.sarif
   - uses: github/codeql-action/upload-sarif@v4
     with:
       sarif_file: mlscent.sarif
   - run: ml_smell_detector analyze src/ --format json --select ML.DATA_LEAKAGE --fail-on-findings