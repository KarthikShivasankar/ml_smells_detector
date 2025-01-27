import pytest
from ml_code_smell_detector import HuggingFaceSmellDetector

def test_huggingface_detector_initialization():
    detector = HuggingFaceSmellDetector()
    assert isinstance(detector, HuggingFaceSmellDetector)
    assert len(detector.smells) == 0

def test_generate_report():
    detector = HuggingFaceSmellDetector()
    detector.smells = [{'smell': 'Test Smell', 'how_to_fix': 'Fix it', 'benefits': 'Better code', 'strategies': 'Do this', 'line_number': 1, 'code_snippet': 'test code', 'file_path': 'test_file.py'}]
    report = detector.generate_report()
    assert 'Test Smell' in report
    assert 'Fix it' in report
    assert 'Better code' in report
    assert 'Do this' in report

