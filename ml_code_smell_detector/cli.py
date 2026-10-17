"""Command line interface. ``ml_smell_detector analyze PATH... [--format text|json|sarif]`` and ``ml_smell_detector rules``.

Exit codes: 0 success, 1 findings reported with --fail-on-findings, 2 usage error (e.g. a path that does not exist).
"""

import argparse
import csv
import json
import os
import sys

from tqdm import tqdm

from ml_code_smell_detector import api
from ml_code_smell_detector.utils import ensure_directory_exists

# Section titles of the text/CSV reports, per detector
SECTIONS = {"framework": "Framework-Specific", "huggingface": "Hugging Face", "general": "General ML"}


def to_sections(result):
    """An api.analyze_file result in the report layout: {section: [{framework, name, fix, benefits, location}]}."""
    sections = {title: [] for title in SECTIONS.values()}
    for f in result["findings"]:
        sections[SECTIONS[f["detector"]]].append({
            "framework": f["framework"],
            # General ML findings are sentences; the report has always shown the whole sentence
            "name": f["message"] if f["detector"] == "general" else f["name"],
            "fix": f["how_to_fix"] or "Not specified",
            "benefits": f["benefits"] or "Not specified",
            "location": f"Line {f['line']}" if f["line"] else "",
        })
    return sections


def analyze_file(file_path, select=None, ignore=None):
    return to_sections(api.analyze_file(file_path, select=select, ignore=ignore))


def write_txt_report(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
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
    with open(output_file, "w", newline="", encoding="utf-8") as f:
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
    """Recursively collect .py files, skipping ignored directories (and virtualenvs, caches, VCS metadata)."""
    return api.collect_python_files(path, ignored_dirs)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="ml_smell_detector",
        description="Static analysis of Python ML code (Pandas, NumPy, scikit-learn, PyTorch, TensorFlow, Hugging Face).",
        epilog="Exit codes: 0 ok, 1 findings with --fail-on-findings, 2 usage error. "
               "Silence a finding in source with '# mlscent: ignore[RULE_ID]' or '# mlscent: ignore-file[RULE_ID]'.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {api.__version__}")
    parser.add_argument("action", choices=["analyze", "rules"], help="analyze PATH..., or list all rule ids as JSON")
    parser.add_argument("path", nargs="*", help="Files or directories to analyze; '-' reads source from stdin")
    parser.add_argument(
        "--format",
        choices=["text", "json", "sarif"],
        default="text",
        help="text: write .txt/.csv reports to --output-dir; json / sarif: print one document to stdout",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to store output files (text format)")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="DIR",
        help="Directory names to ignore during analysis (e.g. --ignore tests docs); .venv, .git, caches etc. always are",
    )
    parser.add_argument("--select", nargs="+", metavar="RULE", help="Only report rule ids starting with these prefixes")
    parser.add_argument("--ignore-rules", nargs="+", metavar="RULE", help="Drop rule ids starting with these prefixes")
    parser.add_argument("--stdin-filename", default="stdin.py", help="File name to analyse stdin ('-') as")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit with status 1 if anything is reported")
    return parser


def _run_analysis(args, parser):
    if not args.path:
        parser.error("analyze requires at least one path")
    results, files = [], []
    for path in args.path:
        if path == "-":
            results.append(api.analyze_source(sys.stdin.read(), filename=args.stdin_filename,
                                              select=args.select, ignore=args.ignore_rules))
        elif os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            files += collect_python_files(path, set(args.ignore))
        else:
            print(f"Error: {path} is not a valid file or directory", file=sys.stderr)
            sys.exit(2)
    files = list(dict.fromkeys(files))
    progress = tqdm(files, desc="Analyzing files", unit="file") if args.format == "text" else files
    results += [api.analyze_file(f, select=args.select, ignore=args.ignore_rules) for f in progress]
    return api.report(results)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.action == "rules":
        print(json.dumps(api.list_rules(), indent=2))
        return

    doc = _run_analysis(args, parser)
    if args.format == "json":
        print(json.dumps(doc, indent=2))
    elif args.format == "sarif":
        print(json.dumps(api.to_sarif(doc), indent=2))
    else:
        ensure_directory_exists(args.output_dir)
        txt_output = os.path.join(args.output_dir, "analysis_report.txt")
        csv_output = os.path.join(args.output_dir, "analysis_report.csv")
        results = {r["file"]: to_sections(r) for r in doc["files"]}
        write_txt_report(results, txt_output)
        write_csv_report(results, csv_output)
        summary = doc["summary"]
        print(f"Analysis complete: {summary['findings']} findings in {summary['files']} files. "
              f"Results written to {txt_output} and {csv_output}")

    if args.fail_on_findings and doc["summary"]["findings"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
