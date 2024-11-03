import pytest
from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector
import astroid

class CrossValidationDetector:
    def __init__(self):
        self.framework_detector = FrameworkSpecificSmellDetector()

    def detect_smells(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            # Directly use the Scikit-learn cross validation detection method
            self.framework_detector.detect_cross_validation(module, file_path)
            
            # Filter only Cross Validation related smells
            cv_smells = [
                smell for smell in self.framework_detector.smells 
                if smell['name'] == 'Cross Validation Checker'
            ]
            
            return cv_smells
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

def test_missing_cross_validation_sample1():
    detector = CrossValidationDetector()
    smells = detector.detect_smells("evaluation/cross_validation/sample1.py")
    assert len(smells) > 0, "Should detect missing cross-validation in sample1"
    print_smell_details(smells)

def test_missing_cross_validation_sample2():
    detector = CrossValidationDetector()
    smells = detector.detect_smells("evaluation/cross_validation/sample2.py")
    assert len(smells) > 0, "Should detect missing cross-validation in sample2"
    print_smell_details(smells)

def test_all_cv_files():
    detector = CrossValidationDetector()
    test_files = [
        "evaluation/cross_validation/sample1.py",
        "evaluation/cross_validation/sample2.py"
    ]
    
    all_smells = []
    for file_path in test_files:
        smells = detector.detect_smells(file_path)
        all_smells.extend(smells)
        
    print("\nAll detected Cross Validation smells:")
    print_smell_details(all_smells)
    
    assert len(all_smells) >= len(test_files), "Should detect Cross Validation issues in all sample files"

def print_smell_details(smells):
    for smell in smells:
        print(f"\nFile: {smell['file_path']}")
        print(f"Line: {smell['line_number']}")
        print(f"How to fix: {smell['how_to_fix']}")
        print(f"Benefits: {smell['benefits']}")
        print(f"Code: {smell['code_snippet']}") 