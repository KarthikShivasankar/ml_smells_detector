# ML Code Smell Detector

ML Code Smell Detector is a Python package that helps identify potential issues and bad practices in machine learning code. It includes detectors for framework-specific smells, Hugging Face-related smells, and general machine learning smells.

## Features

### Framework-Specific Smell Detector
- Detects smells related to Pandas, NumPy, Scikit-learn, TensorFlow, and PyTorch
- Checks for issues like unnecessary iteration, NaN equality checks, and more
- Provides specific advice for each detected smell

### Hugging Face Smell Detector
- Identifies best practices for using the Hugging Face Transformers library
- Checks for model versioning, tokenizer caching, and other Hugging Face-specific issues
- Offers suggestions for improving Hugging Face model usage

### General ML Smell Detector
- Detects common machine learning code smells
- Identifies potential data leakage, magic numbers, and cross-validation issues
- Checks for proper feature scaling, handling of imbalanced datasets, and more

## Installation

You can install the ML Code Smell Detector using pip:

```bash
pip install -e .
```

## Usage

```bash
ml_smell_detector analyze  <path_to_code>
```


## Detailed Feature List

### Framework-Specific Smell Detector
1. Pandas:
   - Unnecessary iteration
   - Chain indexing
   - Inefficient merge operations
   - Inplace operations
   - Inefficient DataFrame conversion
   - Missing data type specifications
2. NumPy:
   - NaN equality checks
   - Missing random seed setting
3. Scikit-learn:
   - Missing feature scaling
   - Absence of pipelines
   - Lack of cross-validation
   - Inconsistent random state usage
   - Missing verbose mode in long-running operations
   - Overreliance on accuracy metric
4. TensorFlow:
   - Missing random seed setting
   - Absence of early stopping
   - Lack of checkpointing
   - Inefficient memory management
   - Missing logging and visualization
5. PyTorch:
   - Missing random seed setting
   - Inefficient data loading
   - Incorrect gradient clearing
   - Missing batch normalization
   - Lack of learning rate scheduling

### Hugging Face Smell Detector
1. Model versioning issues
2. Missing tokenizer and model caching
3. Inconsistent tokenization settings
4. Inefficient data loading practices
5. Lack of distributed training configuration
6. Missing mixed precision training
7. Absence of gradient accumulation for large batches
8. Lack of learning rate scheduling
9. Missing early stopping implementation

### General ML Smell Detector
1. Data leakage detection
2. Magic number usage
3. Inconsistent feature scaling
4. Missing cross-validation
5. Imbalanced dataset handling
6. Feature selection issues
7. Overreliance on single metrics
8. Lack of model persistence
9. Missing reproducibility measures
10. Inefficient data loading for large datasets
11. Unused feature detection
12. Overfitting-prone practices
13. Lack of error handling
14. Hardcoded file paths
15. Missing or incomplete documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.


## Documentation

To build the documentation:

1. Install the development dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Navigate to the `docs` directory:
   ```bash
   cd docs
   ```

3. Build the documentation:
   ```bash
   make html
   ```

4. Open `docs/build/html/index.html` in your web browser to view the documentation.


4. Change to the `docs` directory:

   .. code-block:: console

      cd docs

5. Run the Sphinx build command:

   .. code-block:: console

      sphinx-build -b html source build/html

This will generate the HTML documentation in the `docs/build/html` directory. You can open the `index.html` file in this directory to view the updated documentation.
