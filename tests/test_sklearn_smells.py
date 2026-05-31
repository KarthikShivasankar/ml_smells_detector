"""Tests for Scikit-learn-specific smell detection in FrameworkSpecificSmellDetector."""

import tempfile
import textwrap
from pathlib import Path

from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector


def detect(tmp_py, code):
    detector = FrameworkSpecificSmellDetector()
    path = tmp_py(code)
    detector.detect_smells(path)
    return detector.smells


def names(smells):
    return [s["name"] for s in smells]


# ---------------------------------------------------------------------------
# Scaler Missing Checker
# ---------------------------------------------------------------------------

class TestScalerMissingChecker:
    def test_detects_svm_without_scaler(self, tmp_py):
        # Detector checks for 'SVR' (not 'SVC') in call.func.as_string()
        code = """\
            from sklearn.svm import SVR
            from sklearn.datasets import make_regression
            X, y = make_regression()
            clf = SVR()
            clf.fit(X, y)
        """
        smells = detect(tmp_py, code)
        assert "Scaler Missing Checker" in names(smells)

    def test_no_smell_svm_with_scaler(self, tmp_py):
        code = """\
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.datasets import make_classification
            X, y = make_classification()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = SVC()
            clf.fit(X_scaled, y)
        """
        smells = detect(tmp_py, code)
        assert "Scaler Missing Checker" not in names(smells)

    def test_detects_logistic_regression_without_scaler(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
        """
        smells = detect(tmp_py, code)
        assert "Scaler Missing Checker" in names(smells)


# ---------------------------------------------------------------------------
# Pipeline Checker
# ---------------------------------------------------------------------------

class TestPipelineChecker:
    def test_detects_missing_pipeline(self, tmp_py):
        code = """\
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform([[1, 2], [3, 4]])
            clf = LogisticRegression()
            clf.fit(X_scaled, [0, 1])
        """
        smells = detect(tmp_py, code)
        assert "Pipeline Checker" in names(smells)

    def test_no_smell_with_pipeline(self, tmp_py):
        code = """\
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
            pipe.fit([[1, 2], [3, 4]], [0, 1])
        """
        smells = detect(tmp_py, code)
        assert "Pipeline Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Cross Validation Checker
# ---------------------------------------------------------------------------

class TestCrossValidationChecker:
    def test_detects_missing_cv(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
            preds = clf.predict([[1, 2]])
        """
        smells = detect(tmp_py, code)
        assert "Cross Validation Checker" in names(smells)

    def test_no_smell_with_cross_val_score(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            clf = LogisticRegression()
            scores = cross_val_score(clf, [[1, 2], [3, 4]], [0, 1], cv=2)
        """
        smells = detect(tmp_py, code)
        assert "Cross Validation Checker" not in names(smells)

    def test_no_smell_with_kfold(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import KFold
            clf = LogisticRegression()
            kf = KFold(n_splits=5)
            clf.fit([[1, 2], [3, 4]], [0, 1])
        """
        smells = detect(tmp_py, code)
        assert "Cross Validation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Randomness Control Checker (random_state missing)
# ---------------------------------------------------------------------------

class TestRandomnessControlChecker:
    def test_detects_train_test_split_without_random_state(self, tmp_py):
        code = """\
            from sklearn.model_selection import train_test_split
            X = [[1, 2], [3, 4], [5, 6]]
            y = [0, 1, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" in names(smells)

    def test_no_smell_with_random_state(self, tmp_py):
        code = """\
            from sklearn.model_selection import train_test_split
            X = [[1, 2], [3, 4], [5, 6]]
            y = [0, 1, 0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Verbose Mode Checker
# ---------------------------------------------------------------------------

class TestVerboseModeChecker:
    def test_detects_grid_search_without_verbose(self, tmp_py):
        code = """\
            from sklearn.model_selection import GridSearchCV
            from sklearn.svm import SVC
            param_grid = {'C': [1, 10]}
            gs = GridSearchCV(SVC(), param_grid)
        """
        smells = detect(tmp_py, code)
        assert "Verbose Mode Checker" in names(smells)

    def test_no_smell_with_verbose(self, tmp_py):
        code = """\
            from sklearn.model_selection import GridSearchCV
            from sklearn.svm import SVC
            param_grid = {'C': [1, 10]}
            gs = GridSearchCV(SVC(), param_grid, verbose=2)
        """
        smells = detect(tmp_py, code)
        assert "Verbose Mode Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Dependent Threshold Checker (only accuracy_score, no AUC etc.)
# ---------------------------------------------------------------------------

class TestDependentThresholdChecker:
    def test_detects_only_accuracy_score(self, tmp_py):
        code = """\
            from sklearn.metrics import accuracy_score
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
            preds = clf.predict([[1, 2]])
            score = accuracy_score([0], preds)
        """
        smells = detect(tmp_py, code)
        assert "Dependent Threshold Checker" in names(smells)

    def test_no_smell_with_roc_auc(self, tmp_py):
        code = """\
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
            preds = clf.predict_proba([[1, 2]])[:, 1]
            auc = roc_auc_score([0], preds)
        """
        smells = detect(tmp_py, code)
        assert "Dependent Threshold Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Unit Testing Checker
# ---------------------------------------------------------------------------

class TestUnitTestingChecker:
    def test_detects_missing_tests_in_ml_file(self):
        # pytest temp dirs always contain 'pytest' (has 'test' as substring) which
        # would trigger the is_test_file check. Write to system temp directly.
        code = textwrap.dedent("""\
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
            score = clf.score([[1, 2], [3, 4]], [0, 1])
        """)
        base = Path(tempfile.gettempdir()) / "sklearn_ml_work"
        base.mkdir(exist_ok=True)
        path = str(base / "ml_pipeline.py")
        Path(path).write_text(code)
        detector = FrameworkSpecificSmellDetector()
        detector.detect_smells(path)
        assert "Unit Testing Checker" in names(detector.smells)

    def test_no_smell_when_pytest_present(self, tmp_py):
        code = """\
            import pytest
            from sklearn.linear_model import LogisticRegression

            def test_model():
                clf = LogisticRegression()
                clf.fit([[1, 2], [3, 4]], [0, 1])
                assert clf.score([[1, 2], [3, 4]], [0, 1]) == 1.0
        """
        # Write to a non-test filename to avoid the is_test_file short-circuit
        smells = detect(tmp_py, code)
        assert "Unit Testing Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Data Leakage Checker (preprocessing + training without split)
# ---------------------------------------------------------------------------

class TestDataLeakageChecker:
    def test_detects_fit_transform_without_split(self, tmp_py):
        code = """\
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform([[1, 2], [3, 4]])
            model = LinearRegression()
            model.fit(X_scaled, [1, 2])
        """
        smells = detect(tmp_py, code)
        assert "Data Leakage Checker" in names(smells)

    def test_no_smell_with_split(self, tmp_py):
        code = """\
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            X = [[1, 2], [3, 4], [5, 6]]
            y = [1, 2, 3]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_train_s, y_train)
        """
        smells = detect(tmp_py, code)
        assert "Data Leakage Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Exception Handling Checker
# ---------------------------------------------------------------------------

class TestExceptionHandlingChecker:
    def test_detects_missing_exception_handling(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            clf.fit([[1, 2], [3, 4]], [0, 1])
        """
        smells = detect(tmp_py, code)
        assert "Exception Handling Checker" in names(smells)

    def test_no_smell_with_try_except(self, tmp_py):
        code = """\
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
            try:
                clf.fit([[1, 2], [3, 4]], [0, 1])
            except Exception as e:
                print(e)
        """
        smells = detect(tmp_py, code)
        assert "Exception Handling Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Smell result structure
# ---------------------------------------------------------------------------

class TestSklearnResultStructure:
    def test_framework_label_is_scikitlearn(self, tmp_py):
        code = """\
            from sklearn.svm import SVC
            clf = SVC()
            clf.fit([[1, 2], [3, 4]], [0, 1])
        """
        smells = detect(tmp_py, code)
        sklearn_smells = [s for s in smells if s["framework"] == "ScikitLearn"]
        assert len(sklearn_smells) > 0
