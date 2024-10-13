import pytest
from ml_code_smell_detector import HuggingFaceSmellDetector

def test_huggingface_detector_initialization():
    detector = HuggingFaceSmellDetector()
    assert isinstance(detector, HuggingFaceSmellDetector)
    assert len(detector.smells) == 0

def test_huggingface_detector_detect_smells(tmp_path):
    detector = HuggingFaceSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from transformers import AutoModel\nmodel = AutoModel.from_pretrained('bert-base-uncased')")
    smells = detector.detect_smells(str(test_file))
    assert len(smells) > 0

def test_model_versioning_detection(tmp_path):
    detector = HuggingFaceSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from transformers import AutoModel\nmodel = AutoModel.from_pretrained('bert-base-uncased')")
    smells = detector.detect_smells(str(test_file))
    assert any(smell['smell'] == 'Model versioning not specified' for smell in smells)

def test_tokenizer_caching_detection(tmp_path):
    detector = HuggingFaceSmellDetector()
    test_file = tmp_path / "test_file.py"
    test_file.write_text("from transformers import AutoTokenizer\ntokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')")
    smells = detector.detect_smells(str(test_file))
    assert any(smell['smell'] == 'Tokenizer caching not used' for smell in smells)

def test_generate_report():
    detector = HuggingFaceSmellDetector()
    detector.smells = [{'smell': 'Test Smell', 'how_to_fix': 'Fix it', 'benefits': 'Better code', 'strategies': 'Do this', 'line_number': 1, 'code_snippet': 'test code', 'file_path': 'test_file.py'}]
    report = detector.generate_report()
    assert 'Test Smell' in report
    assert 'Fix it' in report
    assert 'Better code' in report
    assert 'Do this' in report

# Add more tests for specific smell detection methods
