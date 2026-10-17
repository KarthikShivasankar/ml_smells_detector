"""Machine-friendly entry points: one finding schema, stable rule ids, no stderr noise.

    from ml_code_smell_detector.api import analyze_file, analyze_source, list_rules

Each analysis returns ``{"file": str, "findings": [finding, ...], "errors": [str, ...]}`` where a
finding is::

    {"rule_id": "ML.SCIKITLEARN.RANDOMNESS_CONTROL_CHECKER", "detector": "framework",
     "framework": "ScikitLearn", "name": "Randomness Control Checker", "message": "...",
     "line": 12, "snippet": "...", "how_to_fix": "...", "benefits": "..."}

``line`` is 0 for file-level findings. Findings are sorted by (line, rule_id).
"""

import contextlib
import io
import os
import re
import tempfile

from .detectors.framework_detector import FrameworkSpecificSmellDetector
from .detectors.huggingface_detector import HuggingFaceSmellDetector
from .detectors.ml_detector import ML_SmellDetector

# General-ML findings are sentences with values baked in; map them to stable ids by prefix.
GENERAL_RULES = {
    "Potential data leakage": "ML.DATA_LEAKAGE",
    "Magic number detected": "ML.MAGIC_NUMBER",
    "Inconsistent scaling methods": "ML.INCONSISTENT_SCALING",
    "Cross-validation not detected": "ML.NO_CROSS_VALIDATION",
    "No imbalanced dataset handling": "ML.NO_IMBALANCE_HANDLING",
    "Feature selection detected without": "ML.UNVALIDATED_FEATURE_SELECTION",
    "Only accuracy metric": "ML.ACCURACY_ONLY",
    "Only MSE": "ML.MSE_ONLY",
    "Model saving detected without preprocessing": "ML.SAVE_WITHOUT_PREPROCESSING",
    "Model saving detected without clear versioning": "ML.SAVE_WITHOUT_VERSIONING",
    "No random seed setting": "ML.MISSING_SEED",
    "Incomplete seed setting": "ML.INCOMPLETE_SEED",
    "Data loading detected without size checks": "ML.UNCHECKED_DATA_LOADING",
    "Potentially unused features": "ML.UNUSED_FEATURES",
    "Feature engineering function detected without": "ML.FEATURE_ENGINEERING_LEAKAGE",
    "No error handling detected in data processing": "ML.NO_ERROR_HANDLING",
    "Hardcoded file path": "ML.HARDCODED_PATH",
    "Missing docstring for function": "ML.MISSING_DOCSTRING",
    "Missing docstring for class": "ML.MISSING_DOCSTRING",
}

HF_RULES = [
    "Model versioning not specified", "Tokenizer caching not used", "Model caching not used",
    "Deterministic tokenization settings not specified", "Efficient data loading not detected",
    "Distributed training not configured", "Mixed precision training not enabled",
    "Gradient accumulation not configured", "Learning rate scheduler not detected", "Early stopping not implemented",
]


def _slug(prefix, name):
    return prefix + "." + re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _general_id(message):
    return next((rid for prefix, rid in GENERAL_RULES.items() if message.startswith(prefix)), "ML.OTHER")


def _finding(detector, raw):
    if detector == "framework":
        name, framework, message = raw["name"], raw["framework"], raw["name"]
        rule_id = _slug("ML." + framework.upper(), name)
    elif detector == "huggingface":
        name = message = raw["smell"]
        framework, rule_id = "Hugging Face", _slug("ML.HF", name)
    else:
        message, framework = raw["smell"], "General ML"
        rule_id = _general_id(message)
        name = message.split(":")[0].split(". ")[0]
    return {
        "rule_id": rule_id, "detector": detector, "framework": framework, "name": name, "message": message,
        "line": raw.get("line_number") or 0, "snippet": raw.get("code_snippet", ""),
        "how_to_fix": raw.get("how_to_fix"), "benefits": raw.get("benefits"),
    }


def analyze_file(path):
    """Run all detectors on one file. Never prints; detector errors are returned in ``errors``."""
    findings, errors = [], []
    for detector, cls in (("framework", FrameworkSpecificSmellDetector),
                          ("huggingface", HuggingFaceSmellDetector),
                          ("general", ML_SmellDetector)):
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            raw = cls().detect_smells(path) or []
        findings += [_finding(detector, r) for r in raw]
        errors += [line for line in captured.getvalue().splitlines() if line and not line.startswith("Skipping")]
    findings.sort(key=lambda f: (f["line"], f["rule_id"]))
    return {"file": path, "findings": findings, "errors": errors}


def analyze_source(source, filename="module.py"):
    """Analyse a source string. It is written under a neutral temporary path, because some general
    checks are skipped when the path contains words like 'data', 'utils' or 'test_'."""
    with tempfile.TemporaryDirectory(prefix="mlscent_") as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        result = analyze_file(path)
    result["file"] = filename
    return result


def list_rules():
    """Every rule id the detectors can emit, with the framework detector's fix/benefit text."""
    rules = [{"rule_id": _slug("ML." + fw.upper(), r["name"]), "detector": "framework", "framework": fw, "name": r["name"],
              "how_to_fix": r["how_to_fix"], "benefits": r["benefits"]}
             for fw, entries in FrameworkSpecificSmellDetector().get_smells().items() for r in entries]
    rules += [{"rule_id": _slug("ML.HF", n), "detector": "huggingface", "framework": "Hugging Face", "name": n,
               "how_to_fix": None, "benefits": None} for n in HF_RULES]
    general = {}
    for prefix, rid in GENERAL_RULES.items():
        general.setdefault(rid, prefix)  # one entry per id
    general["ML.OTHER"] = "Other general ML finding"  # fallback id for unmapped messages
    rules += [{"rule_id": rid, "detector": "general", "framework": "General ML", "name": prefix,
               "how_to_fix": None, "benefits": None} for rid, prefix in general.items()]
    return rules
