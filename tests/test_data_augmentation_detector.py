import pytest
from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector
import astroid

class DataAugmentationDetector:
    def __init__(self):
        self.framework_detector = FrameworkSpecificSmellDetector()

    def detect_smells(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            # Directly use the PyTorch data augmentation detection method
            self.framework_detector.detect_pytorch_augmentation(module, file_path)
            
            # Filter only Data Augmentation related smells
            augmentation_smells = [
                smell for smell in self.framework_detector.smells 
                if smell['name'] == 'Data Augmentation Checker'
            ]
            
            return augmentation_smells
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

def test_missing_data_augmentation_sample1():
    detector = DataAugmentationDetector()
    smells = detector.detect_smells("evaluation/data_augmentation/sample1.py")
    assert len(smells) > 0, "Should detect missing data augmentation in sample1"
    print_smell_details(smells)

def test_missing_data_augmentation_sample2():
    detector = DataAugmentationDetector()
    smells = detector.detect_smells("evaluation/data_augmentation/sample2.py")
    assert len(smells) > 0, "Should detect missing data augmentation in sample2"
    print_smell_details(smells)

def test_missing_data_augmentation_sample3():
    detector = DataAugmentationDetector()
    smells = detector.detect_smells("evaluation/data_augmentation/sample3.py")
    assert len(smells) > 0, "Should detect missing data augmentation in sample3"
    print_smell_details(smells)

def test_missing_data_augmentation_sample4():
    detector = DataAugmentationDetector()
    smells = detector.detect_smells("evaluation/data_augmentation/sample4.py")
    assert len(smells) > 0, "Should detect missing data augmentation in sample4"
    print_smell_details(smells)

def test_missing_data_augmentation_sample5():
    detector = DataAugmentationDetector()
    smells = detector.detect_smells("evaluation/data_augmentation/sample5.py")
    assert len(smells) > 0, "Should detect missing data augmentation in sample5"
    print_smell_details(smells)

def test_all_augmentation_files():
    detector = DataAugmentationDetector()
    test_files = [
        "evaluation/data_augmentation/sample1.py",
        "evaluation/data_augmentation/sample2.py",
        "evaluation/data_augmentation/sample3.py",
        "evaluation/data_augmentation/sample4.py",
        "evaluation/data_augmentation/sample5.py"
    ]
    
    all_smells = []
    for file_path in test_files:
        smells = detector.detect_smells(file_path)
        all_smells.extend(smells)
        
    print("\nAll detected Data Augmentation smells:")
    print_smell_details(all_smells)
    
    assert len(all_smells) >= len(test_files), "Should detect Data Augmentation issues in all sample files"

def print_smell_details(smells):
    for smell in smells:
        print(f"\nFile: {smell['file_path']}")
        print(f"Line: {smell['line_number']}")
        print(f"How to fix: {smell['how_to_fix']}")
        print(f"Benefits: {smell['benefits']}")
        print(f"Code: {smell['code_snippet']}") 