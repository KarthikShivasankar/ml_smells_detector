import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional

CORPUS_FILEPATH = Path(__file__).parent.parent.resolve() / "nvd_data_with_predictions"
OUTPUT_FILEPATH = Path(__file__)


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Load and return JSON data from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {file_path}")
    except IOError:
        print(f"Error reading file: {file_path}")
    return None


def extract_data_from_cve(cve_entry: Dict) -> Dict[str, Any]:
    """Extract relevant data from a CVE entry."""
    cve_data = cve_entry.get("cve", {})
    cve_metadata = cve_data.get("CVE_data_meta", {})

    return {
        "cve_id": cve_metadata.get("ID"),
        "published_date": cve_entry.get("publishedDate"),
        "last_modified_date": cve_entry.get("lastModifiedDate"),
        "cwe_list": extract_cwe_list(cve_entry),
        "predicted_cwe_list": extract_cwe_list(cve_entry, predicted=True),
        **count_tags_in_references(cve_entry),
        **extract_cvss_details(cve_entry, "original"),
        **extract_cvss_details(cve_entry, "predicted"),
    }


def extract_cwe_list(cve_entry: Dict, predicted: bool = False) -> List[str]:
    """Extract CWE list from a CVE entry."""
    problem_type_key = "predicted_problemtype_data" if predicted else "problemtype_data"
    problem_type_data = (
        cve_entry.get("cve", {}).get("problemtype", {}).get(problem_type_key, [])
    )
    if problem_type_data and problem_type_data[0].get("description", []):
        return [
            item["value"]
            for item in problem_type_data[0]["description"]
            if item["value"] != "NVD-CWE-Other"
        ]
    return []


def extract_cvss_details(cve_entry: Dict, cvss_type: str) -> Dict[str, Any]:
    """Extract CVSS details from a CVE entry."""
    impact = cve_entry.get("impact", {})
    metric_key = (
        "predicted_baseMetricV3" if cvss_type == "predicted" else "baseMetricV3"
    )
    cvss_data = impact.get(metric_key, {})
    cvss = cvss_data.get("cvssV3", {}) if cvss_type == "original" else cvss_data

    if not cvss:
        return {}

    prefix = "predicted_" if cvss_type == "predicted" else ""
    return {
        f"{prefix}cvssv3_version": cvss.get("version"),
        f"{prefix}vector_string": cvss.get("vectorString"),
        f"{prefix}attack_vector": cvss.get("attackVector"),
        f"{prefix}attack_complexity": cvss.get("attackComplexity"),
        f"{prefix}privileges_required": cvss.get("privilegesRequired"),
        f"{prefix}user_interaction": cvss.get("userInteraction"),
        f"{prefix}scope": cvss.get("scope"),
        f"{prefix}confidentiality_impact": cvss.get("confidentialityImpact"),
        f"{prefix}integrity_impact": cvss.get("integrityImpact"),
        f"{prefix}availability_impact": cvss.get("availabilityImpact"),
        f"{prefix}base_score": cvss.get("baseScore"),
        f"{prefix}base_severity": cvss.get("baseSeverity"),
        f"{prefix}exploitability_score": cvss_data.get("exploitabilityScore"),
        f"{prefix}impact_score": cvss_data.get("impactScore"),
    }


def count_tags_in_references(cve_entry: Dict) -> Dict[str, int]:
    """Count occurrences of each tag in references."""
    references = (
        cve_entry.get("cve", {}).get("references", {}).get("reference_data", [])
    )
    tag_counts: Dict[str, int] = {}

    for reference in references:
        for tag in reference.get("tags", []):
            ref_tag = "ref_" + tag.lower().replace(" ", "_").replace("/", "_")
            tag_counts[ref_tag] = tag_counts.get(ref_tag, 0) + 1

    return tag_counts


def process_cve_entries(nvd_filepath: Path) -> List[Dict]:
    """Process all CVE entries from JSON files in the given directory."""
    cve_data = []
    for file_path in sorted(nvd_filepath.iterdir()):
        print(f"Processing: {file_path}")
        corpus = load_json_file(file_path)
        if corpus:
            cve_data.extend(
                extract_data_from_cve(cve_entry)
                for cve_entry in corpus.get("CVE_Items", [])
            )
    return cve_data


def main():
    """Main function to process CVE data and save to CSV."""
    try:
        cve_data = process_cve_entries(CORPUS_FILEPATH)
        df = pd.DataFrame(cve_data)
        output_file = "processed_nvd_data_with_predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"Data has been processed and saved to '{output_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
