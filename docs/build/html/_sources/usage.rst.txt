Usage
=====

Command-line Interface
----------------------

To analyze a single Python file:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/file.py

To analyze all Python files in a directory:

.. code-block:: bash

   ml_smell_detector analyze path/to/your/directory

Python API
----------

You can also use the ML Code Smell Detector as a Python library:

.. code-block:: python

   from ml_code_smell_detector import FrameworkSpecificSmellDetector, HuggingFaceSmellDetector, ML_SmellDetector

   # Initialize detectors
   framework_detector = FrameworkSpecificSmellDetector()
   huggingface_detector = HuggingFaceSmellDetector()
   ml_detector = ML_SmellDetector()

   # Analyze a file
   file_path = 'path/to/your/file.py'
   framework_smells = framework_detector.detect_smells(file_path)
   huggingface_smells = huggingface_detector.detect_smells(file_path)
   ml_smells = ml_detector.detect_smells(file_path)

   # Print reports
   print(framework_detector.generate_report())
   print(huggingface_detector.generate_report())
   print("General ML Smells:")
   for smell in ml_smells:
       print(f"- {smell}")

Use Cases
---------

1. Code Review: Use the tool to automatically check for common ML code smells during code reviews.
2. Continuous Integration: Integrate the tool into your CI/CD pipeline to catch potential issues early.
3. Education: Use the tool to teach best practices in ML code writing to students or junior developers.
4. Refactoring: Identify areas of improvement in existing ML codebases for refactoring.
5. Project Onboarding: Quickly assess the quality of ML code in a new project you're joining.