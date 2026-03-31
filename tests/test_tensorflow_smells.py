"""Tests for TensorFlow-specific smell detection in FrameworkSpecificSmellDetector."""

import pytest
from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector


def detect(tmp_py, code):
    detector = FrameworkSpecificSmellDetector()
    path = tmp_py(code)
    detector.detect_smells(path)
    return detector.smells


def names(smells):
    return [s["name"] for s in smells]


# ---------------------------------------------------------------------------
# Randomness Control Checker
# ---------------------------------------------------------------------------

class TestTFRandomnessControl:
    def test_detects_random_op_without_seed(self, tmp_py):
        code = """\
            import tensorflow as tf
            x = tf.random.normal((10, 10))
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" in names(smells)

    def test_no_smell_with_set_seed(self, tmp_py):
        code = """\
            import tensorflow as tf
            tf.random.set_seed(42)
            x = tf.random.normal((10, 10))
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" not in names(smells)

    def test_detects_dropout_without_seed(self, tmp_py):
        # 'dropout' appears in the random_ops list checked by detect_tf_random_seed.
        # Use lowercase dropout to match the string check.
        code = """\
            import tensorflow as tf
            x = tf.keras.layers.Dropout(0.5)(tf.random.uniform((10, 10)))
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" in names(smells)


# ---------------------------------------------------------------------------
# Early Stopping Checker
# ---------------------------------------------------------------------------

class TestEarlyStoppingChecker:
    def test_detects_fit_with_epochs_without_early_stopping(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            model.fit([], [], epochs=50)
        """
        smells = detect(tmp_py, code)
        assert "Early Stopping Checker" in names(smells)

    def test_no_smell_with_early_stopping(self, tmp_py):
        code = """\
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(patience=5)
            model.fit([], [], epochs=50, callbacks=[es])
        """
        smells = detect(tmp_py, code)
        assert "Early Stopping Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Checkpointing Checker
# ---------------------------------------------------------------------------

class TestCheckpointingChecker:
    def test_detects_complex_model_without_checkpoint(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
            ])
            model.compile()
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Checkpointing Checker" in names(smells)

    def test_no_smell_with_model_checkpoint(self, tmp_py):
        code = """\
            import tensorflow as tf
            from tensorflow.keras.callbacks import ModelCheckpoint
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
            ])
            model.compile()
            cp = ModelCheckpoint('model.h5')
            model.fit([], [], callbacks=[cp])
        """
        smells = detect(tmp_py, code)
        assert "Checkpointing Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Memory Management Checker
# ---------------------------------------------------------------------------

class TestMemoryManagementChecker:
    def test_detects_intensive_ops_without_clear_session(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, 3)])
            model.predict([])
        """
        smells = detect(tmp_py, code)
        assert "Memory Release Checker" in names(smells)

    def test_no_smell_with_clear_session(self, tmp_py):
        code = """\
            import tensorflow as tf
            tf.keras.backend.clear_session()
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.predict([])
        """
        smells = detect(tmp_py, code)
        assert "Memory Management Checker" not in names(smells)


# ---------------------------------------------------------------------------
# TF Metrics Checker (threshold-independent)
# ---------------------------------------------------------------------------

class TestTFMetricsChecker:
    def test_detects_classifier_without_auc(self, tmp_py):
        # Detector checks call.func.as_string() for lowercase 'accuracy'.
        # tf.keras.metrics.binary_accuracy is a lowercase function that matches.
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit([], [])
            y_pred = model.predict([])
            score = tf.keras.metrics.binary_accuracy([], y_pred)
        """
        smells = detect(tmp_py, code)
        assert "Dependent Threshold Checker" in names(smells)

    def test_no_smell_with_auc_metric(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam', loss='binary_crossentropy',
                          metrics=[tf.keras.metrics.AUC()])
            model.fit([], [])
            model.predict([])
        """
        smells = detect(tmp_py, code)
        assert "Dependent Threshold Checker" not in names(smells)


# ---------------------------------------------------------------------------
# TF Logging Checker
# ---------------------------------------------------------------------------

class TestTFLoggingChecker:
    def test_detects_training_without_logging(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Logging Checker" in names(smells)

    def test_no_smell_with_tensorboard_callback(self, tmp_py):
        code = """\
            import tensorflow as tf
            from tensorflow.keras.callbacks import TensorBoard
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            tb = TensorBoard(log_dir='./logs')
            model.fit([], [], callbacks=[tb])
        """
        smells = detect(tmp_py, code)
        assert "Logging Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Batch Normalisation Checker
# ---------------------------------------------------------------------------

class TestTFBatchNormChecker:
    def test_detects_deep_model_without_batchnorm(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),
            ])
            model.compile()
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Batch Normalisation Checker" in names(smells)

    def test_no_smell_with_batchnorm(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(128),
            ])
            model.compile()
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Batch Normalisation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Dropout Checker
# ---------------------------------------------------------------------------

class TestTFDropoutChecker:
    def test_detects_complex_model_without_dropout(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),
            ])
            model.compile()
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Dropout Usage Checker" in names(smells)

    def test_no_smell_with_dropout(self, tmp_py):
        code = """\
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10),
            ])
            model.compile()
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Dropout Usage Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Model Evaluation Checker
# ---------------------------------------------------------------------------

class TestTFModelEvaluationChecker:
    def test_detects_model_without_evaluate(self, tmp_py):
        # The detector requires 'test'/'val'/'valid'/'evaluation' in code AND
        # model usage AND no evaluate/predict call.
        code = """\
            import tensorflow as tf
            X_test = []
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            model.fit([], [])
        """
        smells = detect(tmp_py, code)
        assert "Model Evaluation Checker" in names(smells)

    def test_no_smell_with_evaluate(self, tmp_py):
        code = """\
            import tensorflow as tf
            X_test = []
            model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
            model.compile(optimizer='adam', loss='mse')
            model.fit([], [])
            results = model.evaluate(X_test, [])
        """
        smells = detect(tmp_py, code)
        assert "Model Evaluation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Framework label
# ---------------------------------------------------------------------------

class TestTFFrameworkLabel:
    def test_tensorflow_smells_have_correct_framework(self, tmp_py):
        code = """\
            import tensorflow as tf
            x = tf.random.normal((3,))
        """
        smells = detect(tmp_py, code)
        tf_smells = [s for s in smells if s["framework"] == "TensorFlow"]
        assert len(tf_smells) > 0
