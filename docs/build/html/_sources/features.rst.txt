Features
========

The ML Code Smell Detector checks for various code smells across different categories. Here's a detailed breakdown of the smells it detects:

Framework-Specific Smells
-------------------------

General
^^^^^^^
1. **Import Checker**: Ensures standard naming conventions for imported modules (e.g., `import numpy as np`).

Pandas
^^^^^^
1. **Unnecessary Iteration**: Detects use of `.iterrows()` which is often slower than vectorized operations.
2. **DataFrame Iteration Modification**: Identifies modifications to DataFrames during iteration, which can lead to unexpected behavior.
3. **Chain Indexing**: Detects chained indexing, which can lead to performance issues and unexpected behavior.
4. **Datatype Checker**: Ensures explicit data type setting when importing data to prevent automatic type inference issues.
5. **Column Selection Checker**: Encourages selecting necessary columns after importing DataFrames for clarity and performance.
6. **Merge Parameter Checker**: Checks for proper use of parameters in merge operations to prevent data loss.
7. **InPlace Checker**: Discourages use of `inplace=True` to prevent accidental data loss.
8. **DataFrame Conversion Checker**: Encourages use of `.to_numpy()` instead of `.values` for future compatibility.

NumPy
^^^^^
1. **NaN Equality Checker**: Detects improper NaN comparisons and suggests using `np.isnan()`.
2. **Randomness Control Checker**: Checks for proper random seed setting for reproducibility.

Scikit-learn
^^^^^^^^^^^^
1. **Scaler Missing Checker**: Ensures scaling is applied before scale-sensitive operations.
2. **Pipeline Checker**: Encourages use of Pipelines to prevent data leakage.
3. **Cross Validation Checker**: Checks for proper use of cross-validation techniques.
4. **Randomness Control Checker**: Ensures consistent random state setting across estimators.
5. **Verbose Mode Checker**: Encourages use of verbose mode for long-running processes.
6. **Dependent Threshold Checker**: Suggests use of threshold-independent metrics alongside threshold-dependent ones.
7. **Unit Testing Checker**: Checks for presence of unit tests.
8. **Data Leakage Checker**: Ensures proper train-test splitting to prevent data leakage.
9. **Exception Handling Checker**: Checks for proper exception handling in data processing steps.

TensorFlow
^^^^^^^^^^
1. **Randomness Control Checker**: Checks for proper random seed setting.
2. **Early Stopping Checker**: Encourages use of early stopping to prevent overfitting.
3. **Checkpointing Checker**: Ensures model checkpoints are saved during training.
4. **Memory Release Checker**: Checks for proper memory clearing, especially in loops.
5. **Mask Missing Checker**: Ensures proper masking in operations like `tf.math.log`.
6. **Tensor Array Checker**: Encourages use of `tf.TensorArray` for dynamic tensor lists.
7. **Dependent Threshold Checker**: Similar to Scikit-learn's checker.
8. **Logging Checker**: Encourages use of TensorBoard or other logging mechanisms.
9. **Batch Normalisation Checker**: Checks for use of batch normalization layers.
10. **Dropout Usage Checker**: Encourages use of dropout for regularization.
11. **Data Augmentation Checker**: Checks for data augmentation techniques.
12. **Learning Rate Scheduler Checker**: Encourages use of learning rate schedules.
13. **Model Evaluation Checker**: Ensures proper model evaluation practices.
14. **Unit Testing Checker**: Checks for TensorFlow-specific unit tests.
15. **Exception Handling Checker**: Similar to Scikit-learn's checker.

PyTorch
^^^^^^^
1. **Randomness Control Checker**: Checks for proper random seed setting.
2. **Deterministic Algorithm Usage Checker**: Encourages use of deterministic algorithms.
3. **Randomness Control Checker (PyTorch-Dataloader)**: Checks for proper random seed setting in DataLoader.
4. **Mask Missing Checker**: Similar to TensorFlow's checker.
5. **Net Forward Checker**: Discourages direct calls to `net.forward()`.
6. **Gradient Clear Checker**: Ensures gradients are cleared before each backward pass.
7. **Batch Normalisation Checker**: Similar to TensorFlow's checker.
8. **Dropout Usage Checker**: Similar to TensorFlow's checker.
9. **Data Augmentation Checker**: Checks for use of torchvision transforms.
10. **Learning Rate Scheduler Checker**: Similar to TensorFlow's checker.
11. **Logging Checker**: Checks for use of tensorboardX or similar logging tools.
12. **Model Evaluation Checker**: Ensures model is set to evaluation mode when appropriate.
13. **Unit Testing Checker**: Similar to Scikit-learn's checker.
14. **Exception Handling Checker**: Similar to Scikit-learn's checker.

General ML Smells
-----------------
1. **Data Leakage**: Checks for potential data leakage issues.
2. **Magic Numbers**: Identifies hard-coded constants that should be named variables.
3. **Feature Scaling**: Ensures consistent feature scaling across the dataset.
4. **Cross Validation**: Checks for proper use of cross-validation techniques.
5. **Imbalanced Dataset Handling**: Identifies if techniques for handling imbalanced datasets are used.
6. **Feature Selection**: Checks if feature selection is applied with proper validation.
7. **Metric Selection**: Ensures use of appropriate evaluation metrics.
8. **Model Persistence**: Checks for proper model saving practices.
9. **Reproducibility**: Ensures random seeds are set for reproducibility.
10. **Data Loading**: Suggests efficient data loading practices for large datasets.
11. **Unused Features**: Identifies potentially unused features.
12. **Overfit-Prone Practices**: Checks for practices that might lead to overfitting.
13. **Error Handling**: Ensures proper error handling in data processing.
14. **Hardcoded Filepaths**: Identifies hardcoded file paths.
15. **Documentation**: Checks for presence of docstrings and comments.

Hugging Face-Specific Smells
----------------------------
1. **Model Versioning**: Ensures specific model versions are used for reproducibility.
2. **Tokenizer Caching**: Checks if tokenizers are cached to avoid re-downloading.
3. **Model Caching**: Checks if models are cached to avoid re-downloading.
4. **Deterministic Tokenization**: Ensures consistent tokenization settings.
5. **Efficient Data Loading**: Encourages use of efficient data loading techniques.
6. **Distributed Training**: Checks for configuration of distributed training.
7. **Mixed Precision Training**: Encourages use of mixed precision training for performance.
8. **Gradient Accumulation**: Checks for gradient accumulation for large batch sizes.
9. **Learning Rate Scheduling**: Ensures use of learning rate schedulers.
10. **Early Stopping**: Checks for implementation of early stopping.