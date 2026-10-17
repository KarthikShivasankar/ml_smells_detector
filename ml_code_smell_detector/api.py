"""Machine-friendly entry points: one finding schema, stable rule ids, no stderr noise.

    from ml_code_smell_detector.api import analyze_file, analyze_source, analyze_paths, list_rules

``analyze_file`` / ``analyze_source`` return ``{"file": str, "findings": [finding, ...], "errors": [str, ...],
"suppressed": int}`` where a finding is::

    {"rule_id": "ML.SCIKITLEARN.RANDOMNESS_CONTROL_CHECKER", "detector": "framework",
     "framework": "ScikitLearn", "name": "Randomness Control Checker", "message": "...",
     "line": 12, "snippet": "...", "how_to_fix": "...", "benefits": "..."}

``line`` is 0 for file-level findings (a heuristic about the module as a whole). Findings are sorted by
(line, rule_id). ``analyze_paths`` wraps several files in a report with tool/version/schema metadata and a summary.

Findings can be silenced in the analysed source, by rule id or rule-id prefix:

    x = df.values  # mlscent: ignore[ML.PANDAS.DATAFRAME_CONVERSION_CHECKER]
    # mlscent: ignore-file[ML.SCIKITLEARN.UNIT_TESTING_CHECKER, ML.MISSING_DOCSTRING]

``# mlscent: ignore`` without a list silences every rule on that line; ``# mlscent: ignore-file`` the whole file.
"""

import contextlib
import io
import os
import re
import tempfile
from importlib import metadata

from .detectors.framework_detector import FrameworkSpecificSmellDetector
from .detectors.huggingface_detector import HuggingFaceSmellDetector
from .detectors.ml_detector import ML_SmellDetector
from .utils import read_source

try:
    __version__ = metadata.version("ml-code-smell-detector")
except metadata.PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

TOOL_NAME = "ml-code-smell-detector"
SCHEMA_VERSION = 1  # bump when the shape of a finding or report changes incompatibly

# Directory names never worth analysing (virtualenvs, VCS metadata, caches, build output)
DEFAULT_EXCLUDE_DIRS = frozenset({
    '.git', '.hg', '.svn', '.venv', 'venv', 'env', '.env', '__pycache__', 'node_modules', 'build', 'dist', '.eggs',
    '.tox', '.nox', '.mypy_cache', '.pytest_cache', '.ruff_cache', '.ipynb_checkpoints', 'site-packages',
})

DETECTORS = (("framework", FrameworkSpecificSmellDetector),
             ("huggingface", HuggingFaceSmellDetector),
             ("general", ML_SmellDetector))

_SUPPRESSION = re.compile(r'#\s*mlscent:\s*(ignore-file|ignore)\b(?:\[([^\]]*)\])?')


def _slug(prefix, name):
    return prefix + "." + re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def _finding(detector, raw):
    if detector == "framework":
        name, framework, message = raw["name"], raw["framework"], raw["name"]
        rule_id = _slug("ML." + framework.upper(), name)
    elif detector == "huggingface":
        name = message = raw["smell"]
        framework, rule_id = "Hugging Face", _slug("ML.HF", name)
    else:
        message, framework = raw["smell"], "General ML"
        rule_id = raw.get("rule_id") or ML_SmellDetector.rule_for(message)[0]
        name = message.split(":")[0].split(". ")[0]
    return {
        "rule_id": rule_id, "detector": detector, "framework": framework, "name": name, "message": message,
        "line": raw.get("line_number") or 0, "snippet": raw.get("code_snippet", ""),
        "how_to_fix": raw.get("how_to_fix"), "benefits": raw.get("benefits"),
    }


def _matches(rule_id, prefixes):
    return any(rule_id.startswith(p) for p in prefixes)


def _suppressions(source):
    """({line: [rule prefix, ...]}, [file-level rule prefix, ...]). A bare ignore yields '', which matches every rule."""
    lines, whole_file = {}, []
    for lineno, text in enumerate(source.splitlines(), 1):
        for kind, rules in _SUPPRESSION.findall(text):
            prefixes = [r.strip() for r in rules.split(',') if r.strip()] or ['']
            (whole_file if kind == 'ignore-file' else lines.setdefault(lineno, [])).extend(prefixes)
    return lines, whole_file


def _is_suppressed(finding, lines, whole_file):
    return _matches(finding["rule_id"], whole_file + lines.get(finding["line"], []))


def analyze_file(path, select=None, ignore=None):
    """Run all detectors on one file. Never prints; detector errors are returned in ``errors``.

    ``select`` / ``ignore`` are rule-id prefixes (``"ML.PANDAS"``, ``"ML.HF."``, ``"ML.MAGIC_NUMBER"``): only
    selected rules are kept, ignored ones dropped. ``# mlscent: ignore`` comments in the file are honoured.
    """
    findings, errors = [], []
    for detector, cls in DETECTORS:
        captured = io.StringIO()
        with contextlib.redirect_stderr(captured), contextlib.redirect_stdout(captured):
            raw = cls().detect_smells(path) or []
        findings += [_finding(detector, r) for r in raw]
        for line in captured.getvalue().splitlines():
            if not line or line.startswith("Skipping"):
                continue
            if errors and not line.startswith(("Error", "Unexpected")):
                errors[-1] += " " + line.strip()  # continuation of a multi-line message (astroid syntax errors)
            else:
                errors.append(line)
    if select:
        findings = [f for f in findings if _matches(f["rule_id"], select)]
    if ignore:
        findings = [f for f in findings if not _matches(f["rule_id"], ignore)]
    total = len(findings)
    try:
        lines, whole_file = _suppressions(read_source(path))
        findings = [f for f in findings if not _is_suppressed(f, lines, whole_file)]
    except (OSError, SyntaxError, UnicodeDecodeError):
        pass  # unreadable file: the detectors already reported it in errors
    findings.sort(key=lambda f: (f["line"], f["rule_id"]))
    # Each detector reports the same parse error; keep one of each
    return {"file": path, "findings": findings, "errors": list(dict.fromkeys(errors)), "suppressed": total - len(findings)}


def analyze_source(source, filename="module.py", select=None, ignore=None):
    """Analyse a source string. ``filename`` matters: some general checks are skipped for names containing
    words like ``test_``, ``utils`` or ``data``, and the file is analysed as if it had that name."""
    with tempfile.TemporaryDirectory(prefix="mlscent_") as tmp:
        path = os.path.join(tmp, os.path.basename(filename))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        result = analyze_file(path, select=select, ignore=ignore)
    result["file"] = filename
    result["errors"] = [e.replace(path, filename) for e in result["errors"]]
    return result


def collect_python_files(path, exclude_dirs=()):
    """Sorted .py files under ``path``, skipping DEFAULT_EXCLUDE_DIRS, hidden directories and ``exclude_dirs``."""
    excluded = DEFAULT_EXCLUDE_DIRS | set(exclude_dirs)
    python_files = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in excluded and not d.startswith('.')]
        python_files += [os.path.join(root, f) for f in files if f.endswith(".py")]
    return sorted(python_files)


def summarize(results):
    """Totals over a list of analyze_file results: counts of files, findings, errors and findings per rule."""
    by_rule = {}
    for result in results:
        for f in result["findings"]:
            by_rule[f["rule_id"]] = by_rule.get(f["rule_id"], 0) + 1
    return {
        "files": len(results),
        "files_with_findings": sum(1 for r in results if r["findings"]),
        "findings": sum(by_rule.values()),
        "suppressed": sum(r.get("suppressed", 0) for r in results),
        "errors": sum(len(r["errors"]) for r in results),
        "by_rule": dict(sorted(by_rule.items())),
    }


def analyze_paths(paths, exclude_dirs=(), select=None, ignore=None):
    """Analyse files and/or directories. Raises FileNotFoundError for a path that does not exist.

    Returns ``{"tool", "version", "schema_version", "files": [analyze_file result, ...], "summary": {...}}``.
    """
    files = []
    for path in [paths] if isinstance(paths, str) else paths:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            files += collect_python_files(path, exclude_dirs)
        else:
            raise FileNotFoundError(f"{path} is not a valid file or directory")
    results = [analyze_file(f, select=select, ignore=ignore) for f in dict.fromkeys(files)]
    return report(results)


def report(results):
    """Wrap analyze_file/analyze_source results in the versioned report document."""
    return {"tool": TOOL_NAME, "version": __version__, "schema_version": SCHEMA_VERSION,
            "files": results, "summary": summarize(results)}


def list_rules():
    """Every rule id the detectors can emit, with how-to-fix and benefit text."""
    rules = [{"rule_id": _slug("ML." + fw.upper(), r["name"]), "detector": "framework", "framework": fw, "name": r["name"],
              "how_to_fix": r["how_to_fix"], "benefits": r["benefits"]}
             for fw, entries in FrameworkSpecificSmellDetector().get_smells().items() for r in entries]
    rules += [{"rule_id": _slug("ML.HF", name), "detector": "huggingface", "framework": "Hugging Face", "name": name,
               "how_to_fix": fix, "benefits": benefits} for name, (fix, benefits, _) in HuggingFaceSmellDetector.RULES.items()]
    seen = set()
    for prefix, rule_id, fix, benefits in ML_SmellDetector.RULES:
        if rule_id not in seen:  # one entry per id (both docstring messages share ML.MISSING_DOCSTRING)
            seen.add(rule_id)
            rules.append({"rule_id": rule_id, "detector": "general", "framework": "General ML", "name": prefix,
                          "how_to_fix": fix, "benefits": benefits})
    rules.append({"rule_id": "ML.OTHER", "detector": "general", "framework": "General ML",  # fallback for unmapped messages
                  "name": "Other general ML finding", "how_to_fix": None, "benefits": None})
    return rules


def to_sarif(report_doc):
    """Convert an analyze_paths report to SARIF 2.1.0 (GitHub code scanning, IDEs, CI dashboards)."""
    rules = {r["rule_id"]: r for r in list_rules()}
    used = sorted({f["rule_id"] for result in report_doc["files"] for f in result["findings"]})
    index = {rule_id: i for i, rule_id in enumerate(used)}
    results = []
    for result in report_doc["files"]:
        for f in result["findings"]:
            region = {"startLine": f["line"]} if f["line"] else {}
            results.append({
                "ruleId": f["rule_id"], "ruleIndex": index[f["rule_id"]], "level": "warning",
                "message": {"text": f["message"] if not f["how_to_fix"] else f"{f['message']}. {f['how_to_fix']}"},
                "locations": [{"physicalLocation": {"artifactLocation": {"uri": result["file"].replace(os.sep, "/")},
                                                    **({"region": region} if region else {})}}],
            })
    driver_rules = [{
        "id": rule_id, "name": rules.get(rule_id, {}).get("name", rule_id),
        "shortDescription": {"text": rules.get(rule_id, {}).get("name", rule_id)},
        **({"help": {"text": rules[rule_id]["how_to_fix"]}} if rules.get(rule_id, {}).get("how_to_fix") else {}),
    } for rule_id in used]
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0",
        "runs": [{"tool": {"driver": {"name": TOOL_NAME, "version": report_doc["version"],
                                      "informationUri": "https://github.com/KarthikShivasankar/ml_smells_detector",
                                      "rules": driver_rules}},
                  "results": results}],
    }
