import pytest
from ml_code_smell_detector import FrameworkSpecificSmellDetector

def test_framework_detector_initialization():
    detector = FrameworkSpecificSmellDetector()
    assert isinstance(detector, FrameworkSpecificSmellDetector)
    assert len(detector.smells) == 0

def test_generate_report():
    detector = FrameworkSpecificSmellDetector()
    detector.smells = [{'framework': 'Test', 'name': 'Test Smell', 'how_to_fix': 'Fix it', 'benefits': 'Better code', 'strategies': 'Do this', 'line_number': 1, 'code_snippet': 'test code', 'file_path': 'test_file.py'}]
    report = detector.generate_report()
    assert 'Test Smell' in report
    assert 'Fix it' in report
    assert 'Better code' in report
    assert 'Do this' in report

