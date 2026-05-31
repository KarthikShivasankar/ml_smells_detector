import argparse
import csv
import os

from tqdm import tqdm

from ml_code_smell_detector import (
    FrameworkSpecificSmellDetector,
    HuggingFaceSmellDetector,
    ML_SmellDetector,
)
from ml_code_smell_detector.utils import ensure_directory_exists


def analyze_file(file_path):
    framework_detector = FrameworkSpecificSmellDetector()
    huggingface_detector = HuggingFaceSmellDetector()
    ml_detector = ML_SmellDetector()

    framework_detector.detect_smells(file_path)
    huggingface_detector.detect_smells(file_path)
    ml_detector.detect_smells(file_path)

    return {
        "Framework-Specific": framework_detector.get_results(),
        "Hugging Face": huggingface_detector.get_results(),
        "General ML": ml_detector.get_results(),
    }


def write_txt_report(results, output_file):
    with open(output_file, "w") as f:
        for file_path, detectors in results.items():
            f.write(f"Analysis results for {file_path}:\n")
            for detector_name, smells in detectors.items():
                f.write(f"\n{detector_name} Smells:\n")
                smell_counts = {}
                for smell in smells:
                    smell_counts[smell["name"]] = smell_counts.get(smell["name"], 0) + 1
                    f.write(f"- {smell['name']}\n")
                    f.write(f"  Framework: {smell['framework']}\n")
                    f.write(f"  How to fix: {smell['fix']}\n")
                    f.write(f"  Benefits: {smell['benefits']}\n")
                    if smell["location"]:
                        f.write(f"  Location: {smell['location']}\n")
                    f.write("\n")
                f.write("Smell Counts:\n")
                for smell_name, count in smell_counts.items():
                    f.write(f"  {smell_name}: {count}\n")
                f.write(f"Total smells detected: {len(smells)}\n\n")


def write_csv_report(results, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Framework", "Smell/Checker Name", "How to Fix", "Benefits", "File Path", "Location", "Count"]
        )
        for file_path, detectors in results.items():
            for detector_name, smells in detectors.items():
                smell_counts = {}
                for smell in smells:
                    smell_counts[smell["name"]] = smell_counts.get(smell["name"], 0) + 1
                for smell in smells:
                    writer.writerow(
                        [
                            smell["framework"],
                            smell["name"],
                            smell["fix"],
                            smell["benefits"],
                            file_path,
                            smell["location"],
                            smell_counts[smell["name"]],
                        ]
                    )


def collect_python_files(path, ignored_dirs):
    """Recursively collect .py files, skipping any ignored directories."""
    python_files = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def build_arg_parser():
    parser = argparse.ArgumentParser(description="ML Code Smell Detector")
    parser.add_argument("action", choices=["analyze"], help="Action to perform")
    parser.add_argument("path", help="Path to file or directory to analyze")
    parser.add_argument("--output-dir", default="output", help="Directory to store output files")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="DIR",
        help="Directory names to ignore during analysis (e.g. --ignore tests docs)",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_directory_exists(args.output_dir)
    txt_output = os.path.join(args.output_dir, "analysis_report.txt")
    csv_output = os.path.join(args.output_dir, "analysis_report.csv")

    if args.action == "analyze":
        results = {}
        if os.path.isfile(args.path):
            python_files = [args.path]
        elif os.path.isdir(args.path):
            python_files = collect_python_files(args.path, set(args.ignore))
        else:
            print(f"Error: {args.path} is not a valid file or directory")
            return

        for file_path in tqdm(python_files, desc="Analyzing files", unit="file"):
            results[file_path] = analyze_file(file_path)

        write_txt_report(results, txt_output)
        write_csv_report(results, csv_output)
        print(f"Analysis complete. Results written to {txt_output} and {csv_output}")


if __name__ == "__main__":
    main()
