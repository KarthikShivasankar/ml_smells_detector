import argparse
import os
from ml_code_smell_detector import FrameworkSpecificSmellDetector, HuggingFaceSmellDetector, ML_SmellDetector

def analyze_file(file_path):
    framework_detector = FrameworkSpecificSmellDetector()
    huggingface_detector = HuggingFaceSmellDetector()
    ml_detector = ML_SmellDetector()

    framework_detector.detect_smells(file_path)
    huggingface_detector.detect_smells(file_path)
    ml_detector.detect_smells(file_path)

    print(f"Analysis results for {file_path}:")
    print("\nFramework-Specific Smells:")
    print(framework_detector.generate_report())
    print("\nHugging Face Smells:")
    print(huggingface_detector.generate_report())
    print("\nGeneral ML Smells:")
    print(ml_detector.generate_report())

def main():
    parser = argparse.ArgumentParser(description="ML Code Smell Detector")
    parser.add_argument("action", choices=["analyze"], help="Action to perform")
    parser.add_argument("path", help="Path to file or directory to analyze")
    args = parser.parse_args()

    if args.action == "analyze":
        if os.path.isfile(args.path):
            analyze_file(args.path)
        elif os.path.isdir(args.path):
            for root, _, files in os.walk(args.path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        analyze_file(file_path)
        else:
            print(f"Error: {args.path} is not a valid file or directory")

if __name__ == "__main__":
    main()