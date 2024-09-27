import pytest
from ml_code_smell_detector import ML_SmellDetector

def test_ml_detector_initialization():
    detector = ML_SmellDetector()
    assert isinstance(detector, ML_SmellDetector)
    assert len(detector.smells) == 0

def test_ml_detector_detect_smells(tmp_path):
    detector = ML_SmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("import numpy as np\nx = np.array([1, 2, 3])")
    smells = detector.detect_smells(str(test_file))
    assert len(smells) > 0

def test_data_leakage_detection(tmp_path):
    detector = ML_SmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)")
    smells = detector.detect_smells(str(test_file))
    assert "Potential data leakage: Preprocessing applied before train-test split" in smells

def test_magic_numbers_detection(tmp_path):
    detector = ML_SmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("learning_rate = 0.01\nepochs = 100")
    smells = detector.detect_smells(str(test_file))
    assert any("Magic number detected" in smell for smell in smells)

def test_cross_validation_detection(tmp_path):
    detector = ML_SmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")
    smells = detector.detect_smells(str(test_file))
    assert "Cross-validation not detected" in smells

def test_documentation_detection(tmp_path):
    detector = ML_SmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("def train_model(X, y):\n    # Train the model\n    pass")
    smells = detector.detect_smells(str(test_file))
    assert any("Missing docstring" in smell for smell in smells)

# Add more tests for specific smell detection methods