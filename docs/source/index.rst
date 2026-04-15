Welcome to ML Code Smell Detector's documentation!
==================================================

ML Code Smell Detector is a Python package that helps identify potential issues and bad practices in machine learning code. It includes detectors for framework-specific smells, Hugging Face-related smells, and general machine learning smells.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   features
   api
   detectors/index
   contributing
   changelog

Features
--------

- Framework-Specific Smell Detector: Detects smells related to Pandas, NumPy, Scikit-learn, TensorFlow, and PyTorch
- Hugging Face Smell Detector: Identifies best practices for using the Hugging Face Transformers library
- General ML Smell Detector: Detects common machine learning code smells

For a detailed list of all features, please see the :doc:`features` page.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install ml-code-smell-detector

Analyze a Python file:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/file.py

For more detailed usage instructions, see the :doc:`usage` page.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`