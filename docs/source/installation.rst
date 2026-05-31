Installation
============

Install from PyPI:

.. code-block:: bash

   pip install ml-code-smell-detector

For development, clone the repository and install in editable mode with ``uv``:

.. code-block:: bash

   git clone https://github.com/KarthikShivasankar/ml_smells_detector.git
   cd ml_smells_detector
   uv pip install -e ".[dev]"

Or with standard pip:

.. code-block:: bash

   pip install -e ".[dev]"

Requirements
------------

- Python 3.10+

Runtime dependencies (installed automatically):

- ``astroid`` — AST parsing
- ``tqdm`` — progress bars

The tool performs **static analysis only** and does not require ML frameworks
(NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Transformers) to be installed.