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

Options
-------

- ``path`` — path to a ``.py`` file or a directory to analyze.
- ``--output-dir DIR`` — directory to write reports to (default: ``output/``).
- ``--ignore DIR [DIR ...]`` — directory names to skip during analysis.

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
     uses: actions/upload-artifact@v3
     with:
       name: ml-smell-reports
       path: reports/