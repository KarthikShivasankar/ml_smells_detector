import pytest
from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector
import astroid

class BatchNormDetector:
    def __init__(self):
        self.framework_detector = FrameworkSpecificSmellDetector()

    def detect_smells(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            # Directly use the PyTorch batch norm detection method
            self.framework_detector.detect_pytorch_batch_norm(module, file_path)
            
            # Filter only BatchNorm related smells
            batch_norm_smells = [
                smell for smell in self.framework_detector.smells 
                if smell['name'] == 'Batch Normalisation Checker'
            ]
            
            return batch_norm_smells
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

def test_batch_norm_before_activation():
    detector = BatchNormDetector()
    smells = detector.detect_smells("evaluation/batch_norm/sample3.py")
    assert len(smells) > 0, "Should detect BatchNorm before activation anti-pattern"
    print_smell_details(smells)

def test_inconsistent_batch_norm():
    detector = BatchNormDetector()
    smells = detector.detect_smells("evaluation/batch_norm/sample2.py")
    assert len(smells) > 0, "Should detect inconsistent BatchNorm usage"
    print_smell_details(smells)

def test_small_batch_size():
    detector = BatchNormDetector()
    smells = detector.detect_smells("evaluation/batch_norm/sample4.py")
    assert len(smells) > 0, "Should detect BatchNorm with small batch size"
    print_smell_details(smells)

def test_batch_norm_rnn():
    detector = BatchNormDetector()
    smells = detector.detect_smells("evaluation/batch_norm/sample5.py")
    assert len(smells) > 0, "Should detect BatchNorm with RNN anti-pattern"
    print_smell_details(smells)

def test_all_batch_norm_files():
    detector = BatchNormDetector()
    test_files = [
        "evaluation/batch_norm/sample1.py",
        "evaluation/batch_norm/sample2.py",
        "evaluation/batch_norm/sample3.py",
        "evaluation/batch_norm/sample4.py",
        "evaluation/batch_norm/sample5.py"
    ]
    
    all_smells = []
    for file_path in test_files:
        smells = detector.detect_smells(file_path)
        all_smells.extend(smells)
        
    print("\nAll detected BatchNorm smells:")
    print_smell_details(all_smells)
    
    assert len(all_smells) >= len(test_files), "Should detect BatchNorm issues in all sample files"

def print_smell_details(smells):
    for smell in smells:
        print(f"\nFile: {smell['file_path']}")
        print(f"Line: {smell['line_number']}")
        print(f"How to fix: {smell['how_to_fix']}")
        print(f"Benefits: {smell['benefits']}")
        print(f"Code: {smell['code_snippet']}")