import pytest
from ml_code_smell_detector import FrameworkSpecificSmellDetector

def test_framework_detector_initialization():
    detector = FrameworkSpecificSmellDetector()
    assert isinstance(detector, FrameworkSpecificSmellDetector)
    assert len(detector.smells) == 0

def test_framework_detector_detect_smells(tmp_path):
    detector = FrameworkSpecificSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("import pandas as pd\ndf = pd.DataFrame()")
    smells = detector.detect_smells(str(test_file))
    assert len(smells) > 0

def test_pandas_smell_detection(tmp_path):
    detector = FrameworkSpecificSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("import pandas as pd\ndf = pd.DataFrame()\nfor index, row in df.iterrows():\n    print(row)")
    smells = detector.detect_smells(str(test_file))
    assert any(smell['name'] == 'Unnecessary Iteration' for smell in smells)

def test_numpy_smell_detection(tmp_path):
    detector = FrameworkSpecificSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("import numpy as np\nx = np.array([1, 2, 3])\nif x[0] == np.nan:\n    print('NaN')")
    smells = detector.detect_smells(str(test_file))
    assert any(smell['name'] == 'NaN Equality Checker' for smell in smells)

def test_sklearn_smell_detection(tmp_path):
    detector = FrameworkSpecificSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier()")
    smells = detector.detect_smells(str(test_file))
    assert any(smell['name'] == 'Randomness Control Checker' for smell in smells)

def test_generate_report():
    detector = FrameworkSpecificSmellDetector()
    detector.smells = [{'name': 'Test Smell', 'how_to_fix': 'Fix it', 'benefits': 'Better code', 'strategies': 'Do this'}]
    report = detector.generate_report()
    assert 'Test Smell' in report
    assert 'Fix it' in report
    assert 'Better code' in report
    assert 'Do this' in report

# Add more tests for specific smell detection methods