Features
========

The ML Code Smell Detector checks for various code smells across different categories. Here's a detailed breakdown of the smells it detects.
Every rule has a stable id (``ml_smell_detector rules`` lists them with how-to-fix text); findings about one
statement carry its line, file-level heuristics are reported at line 0.

Framework-Specific Smells
-------------------------

General
^^^^^^^
1. **Import Checker**: Ensures standard naming conventions for imported modules (e.g., `import numpy as np`).

Pandas
^^^^^^
1. **Unnecessary Iteration**: Detects use of `.iterrows()` which is often slower than vectorized operations.
2. **DataFrame Iteration Modification**: Writes to the frame being iterated (``df.loc[i, 'a'] = v`` inside ``for i, row in df.iterrows()``) or to the row copy (``row['a'] = v``, which has no effect on ``df``).
3. **Chain Indexing**: Chained indexing, read or write, on a pandas object (``df['a'][0]``, ``df[mask]['a'] = v``); use ``.loc``/``.iloc``. The receiver must look like a DataFrame/Series (annotated, built by a pandas call or derived from one, or a conventional name such as ``df``); nested dicts/lists (``cfg['a']['b']``, ``grid[i][j]``) and arrays converted out of pandas (``X = df.values``) are not reported.
4. **Datatype Checker**: ``pd.read_csv`` without ``dtype=`` in a module that groups, merges or sorts the data.
5. **Column Selection Checker**: A ``pd.read_*`` call without ``usecols=``/``columns=`` whose columns are never narrowed down (``df[['a', 'b']]``) afterwards.
6. **Merge Parameter Checker**: ``pd.merge(a, b)`` or ``a.merge(b)`` without ``how``/``on``/``validate`` (or ``left_on``/``right_on``/``*_index``).
7. **InPlace Checker**: Every ``inplace=True`` call; ``df2 = df.dropna(inplace=True)`` (``df2`` is ``None``) included.
8. **DataFrame Conversion Checker**: Encourages use of `.to_numpy()` instead of `.values` for future compatibility.

NumPy
^^^^^
1. **NaN Equality Checker**: ``x == np.nan`` / ``!=`` on either side, also ``math.nan`` and ``float('nan')``; suggests ``np.isnan()``.
2. **Randomness Control Checker**: Draws from ``np.random.*`` without ``np.random.seed()`` (reported at the first draw), and ``np.random.default_rng()`` / ``RandomState()`` created without a seed.
3. **Array Creation Efficiency**: ``np.array``/``zeros``/``ones``/``empty`` without ``dtype``.
4. **Inefficient Operations**: Reductions inside loops, and arrays grown in a loop (``np.concatenate``/``append``/``vstack``/``hstack``), which is quadratic.
5. **Dtype Consistency**: Arithmetic mixing numpy int and float scalar types.
6. **Broadcasting Risk**: Arithmetic between reshaped/transposed arrays.
7. **Copy-View Confusion**: ``view = arr[a:b]`` (basic slicing returns a view) followed by a write to ``view``, which also modifies ``arr``.
8. **Missing Axis Specification**: numpy reductions (``sum``, ``mean``, ``max``, ``argmax``, ``std``, ...) called without ``axis``.

Scikit-learn
^^^^^^^^^^^^
1. **Scaler Missing Checker**: A scale-sensitive estimator (SVMs, k-NN, k-means, PCA, linear models, MLPs, ...) in a module that never scales features.
2. **Pipeline Checker**: Preprocessing and model steps without ``Pipeline``/``make_pipeline``.
3. **Cross Validation Checker**: Checks for proper use of cross-validation techniques.
4. **Randomness Control Checker**: Ensures consistent random state setting across estimators.
5. **Verbose Mode Checker**: Encourages use of verbose mode for long-running processes.
6. **Dependent Threshold Checker**: Suggests use of threshold-independent metrics alongside threshold-dependent ones.
7. **Unit Testing Checker**: Checks that a test file (`test_<module>.py` or `<module>_test.py`, next to the module or in a `tests`/`test` directory) exists.
8. **Data Leakage Checker**: Ensures proper train-test splitting to prevent data leakage.
9. **Exception Handling Checker**: Checks that model persistence I/O (`joblib`/`pickle` `load`/`dump`) is guarded by exception handling.

TensorFlow
^^^^^^^^^^
1. **Randomness Control Checker**: Checks for proper random seed setting.
2. **Early Stopping Checker**: Encourages use of early stopping to prevent overfitting.
3. **Checkpointing Checker**: Ensures model checkpoints are saved during training.
4. **Memory Release Checker**: Models built or trained inside a loop without ``tf.keras.backend.clear_session()``.
5. **Mask Missing Checker**: Ensures proper masking in operations like `tf.math.log`.
6. **Tensor Array Checker**: A Python list grown in a loop inside a ``@tf.function``; use ``tf.TensorArray``.
7. **Dependent Threshold Checker**: Similar to Scikit-learn's checker.
8. **Logging Checker**: Encourages use of TensorBoard or other logging mechanisms.
9. **Batch Normalisation Checker**: Checks for use of batch normalization layers.
10. **Dropout Usage Checker**: Encourages use of dropout for regularization.
11. **Data Augmentation Checker**: Checks for data augmentation techniques.
12. **Learning Rate Scheduler Checker**: Encourages use of learning rate schedules.
13. **Model Evaluation Checker**: Ensures proper model evaluation practices.
14. **Unit Testing Checker**: Checks that a test file exists for a module that trains or evaluates a model.
15. **Exception Handling Checker**: ``tf.keras.models.load_model`` / ``tf.saved_model.load|save`` outside ``try``.

PyTorch
^^^^^^^
1. **Randomness Control Checker**: Checks for proper random seed setting.
2. **Deterministic Algorithm Usage Checker**: Encourages use of deterministic algorithms.
3. **Randomness Control Checker (PyTorch-Dataloader)**: Checks for proper random seed setting in DataLoader.
4. **Mask Missing Checker**: Similar to TensorFlow's checker.
5. **Net Forward Checker**: Any ``x.forward(...)`` call, which bypasses hooks; ``super().forward(...)`` is fine.
6. **Gradient Clear Checker**: ``loss.backward()`` with an optimizer but no ``zero_grad()``.
7. **Batch Normalisation Checker**: Similar to TensorFlow's checker.
8. **Dropout Usage Checker**: Similar to TensorFlow's checker.
9. **Data Augmentation Checker**: Checks for use of torchvision transforms.
10. **Learning Rate Scheduler Checker**: Similar to TensorFlow's checker.
11. **Logging Checker**: Checks for use of tensorboardX or similar logging tools.
12. **Model Evaluation Checker**: Ensures model is set to evaluation mode when appropriate.

General ML Smells
-----------------
1. **Data Leakage**: Checks for potential data leakage issues.
2. **Magic Numbers**: Unnamed numeric literals in comparisons and arithmetic (``if p > 0.87``, ``x / 255``). Binding a literal to a name is the fix, so ``threshold = 0.75`` is not reported.
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
14. **Hardcoded Filepaths**: Absolute/relative paths and ``dir/file.ext`` strings; URLs, MIME types, Hub ids and fragments joined onto a base (``root + '/x.csv'``) are not reported.
15. **Documentation**: Checks for presence of docstrings and comments.

Hugging Face-Specific Smells
----------------------------
1. **Model Versioning**: Ensures specific model versions are used for reproducibility.
2. **Tokenizer Caching**: Flags tokenizer loads that bypass the Hugging Face cache (`force_download=True`).
3. **Model Caching**: Flags model loads that bypass the Hugging Face cache (`force_download=True`).
4. **Deterministic Tokenization**: Ensures consistent tokenization settings.
5. **Efficient Data Loading**: Encourages use of efficient data loading techniques.
6. **Distributed Training**: Checks for configuration of distributed training.
7. **Mixed Precision Training**: Encourages use of mixed precision training for performance.
8. **Gradient Accumulation**: Checks for gradient accumulation for large batch sizes.
9. **Learning Rate Scheduling**: A custom training loop (``loss.backward()`` + ``optimizer.step()``) without a scheduler. ``Trainer`` is not reported: it always applies ``lr_scheduler_type`` (default ``linear``).
10. **Early Stopping**: Checks for implementation of early stopping.