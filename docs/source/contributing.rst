Contributing
============

Contributions are welcome! Please follow the steps below.

Setting Up
----------

.. code-block:: bash

   git clone https://github.com/KarthikShivasankar/ml_smells_detector.git
   cd ml_smells_detector
   uv pip install -e ".[dev]"

Running Tests
-------------

.. code-block:: bash

   uv run python -m pytest tests/

Adding a Detector
-----------------

1. Add detection logic to the appropriate detector class in ``ml_code_smell_detector/detectors/``.
2. Each smell dict must include the keys: ``name``, ``framework``, ``fix``, ``benefits``, ``location``.
3. Add tests under ``tests/`` covering both detection and non-detection cases.
4. Update ``docs/source/features.rst`` with the new smell description.

Code Style
----------

- Max line length: 150 characters (configured in ``pyproject.toml``).
- Run ``flake8`` before submitting: ``uv run flake8 ml_code_smell_detector/``.

Submitting Changes
------------------

1. Fork the repository and create a feature branch.
2. Ensure all tests pass.
3. Open a pull request with a clear description of the change.
