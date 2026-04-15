import pandas as pd
import json, os, datetime, numbers, csv
import numpy as np
from .config_reader import ConfigParserEPSS

def get_nvd_feature_column_names(features: dict) -> list:
    columns = []
    for f, v in features.items():
        if not v:
            continue
        if isinstance(v, list):
            for item in v:
                name = f + "_" + str(item)
                columns.append(name)
        else:
            columns.append(f)
    return columns

def read_json_file(filepath: str) -> dict:
    with open(filepath, "r") as f:
        corpus = json.load(f)
    return corpus

def read_NVD_directory(config: ConfigParserEPSS) -> pd.DataFrame:
    df = pd.DataFrame()
    columns = ["cve"] + get_nvd_feature_column_names(config.nvd_features)
    column2list = {c: [] for c in columns}
    for f in os.listdir(config.nvd_filepath):
        filepath = config.nvd_filepath + os.sep + f
        print(filepath)
        nvd_dict = read_json_file(filepath)
        for entry in nvd_dict["CVE_Items"]:
            for c in columns:
                column2list[c].append(FEATURE2FUNCTION[c](entry, config))
    for c, l in column2list.items():
        df[c] = l
    return df

def get_missing_nvd_features(df: pd.DataFrame, config: ConfigParserEPSS) -> dict|bool:
    print("Checking all selected features are present in NVD feature dataframe.")
    columns = set(df.columns)
    features = set(get_nvd_feature_column_names(config.nvd_features))
    missing = list(features.difference(columns))
    if missing:
        return {m: True for m in missing}
    else:
        return False

    
def add_missing_nvd_features(df: pd.DataFrame, features: dict, config: ConfigParserEPSS) -> pd.DataFrame:
    cve2features = {}
    columns = get_nvd_feature_column_names(features)
    column2list = {c: [] for c in columns}
    for f in os.listdir(config.nvd_filepath):
        filepath = config.nvd_filepath + os.sep + f
        print(filepath)
        nvd_dict = read_json_file(filepath)
        for entry in nvd_dict["CVE_Items"]:
            cve = get_cve_id(entry, config)
            cve2features[cve] = {}
            for c in columns:
                cve2features[cve][c] = FEATURE2FUNCTION[c](entry, config)
    ## ensure correct values attributed to correct CVE entry
    for i, r in df.iterrows():
        for c in columns:
            try:
                column2list[c].append(cve2features[r["cve"]][c])
            except:
                data_type = type(column2list[c][-1])
                column2list[c].append(type(""))
    for c, l in column2list.items():
        df[c] = l
    return df

def is_cve_missing(df: pd.DataFrame, config: ConfigParserEPSS) -> bool:
    pass

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
def get_age(entry: dict, config: ConfigParserEPSS) -> int:
    try:
        published_date_string = entry["publishedDate"]
    except:
        return -99
    published_date_components = [int(c) for c in published_date_string.split("T")[0].split("-")]
    published_date_time = [int(c) for c in published_date_string[:-1].split("T")[1].split(":")]
    published_date = datetime.datetime(published_date_components[0],
                                       published_date_components[1],
                                       published_date_components[2],
                                       published_date_time[0],
                                       published_date_time[1])
    return (config.date - published_date).days 

### redundant config argument so as to use list comprehension in "read_NVD_directory" and "add_missing_nvd_features"

def get_cvss_exploitability_score(entry: dict, config: ConfigParserEPSS) -> float:
    if "baseMetricV3" in entry["impact"].keys():
        return entry["impact"]["baseMetricV3"]["exploitabilityScore"]
    if "baseMetricV2" in entry["impact"].keys():
        return entry["impact"]["baseMetricV2"]["exploitabilityScore"]
    else:
        return -99.

def get_cvss_impact_score(entry: dict, config: ConfigParserEPSS) -> float:
    if "baseMetricV3" in entry["impact"].keys():
        return entry["impact"]["baseMetricV3"]["impactScore"]
    if "baseMetricV2" in entry["impact"].keys():
        return entry["impact"]["baseMetricV2"]["impactScore"]
    else:
        return -99.
    

def get_cve_id(entry: dict, config: ConfigParserEPSS) -> str:
    try:
        return entry["cve"]["CVE_data_meta"]["ID"]
    except:
        return "NVD-CVE-noinfo"

def get_cwe_id(entry: dict, config: ConfigParserEPSS):
    try:
        return entry["cve"]["problemtype"]["problemtype_data"][0]["description"][0]["value"]
    except:
        return "NVD-CW-noinfo"
    
    
def get_n_references(entry: dict, config: ConfigParserEPSS) -> int:
    try:
        return len(entry["cve"]["references"]["reference_data"])
    except:
        return 0


def get_description(entry: dict, config: ConfigParserEPSS) -> str:
    try:
        return entry["cve"]["description"]["description_data"][0]["value"]
    except:
        return ""
    



def get_cve_vector(entry, config):
    try:
        return entry["impact"]["baseMetricV3"]["cvssV3"]["vectorString"]
    except KeyError:
        return "NVD-CVE-vector-noinfo"

def count_tag_entries(entry, config, tag_name):
    references = entry.get("cve", {}).get("references", {}).get("reference_data", [])
    return sum(tag_name in reference.get("tags", []) for reference in references)

def count_exploit_entries(entry, config):
    return count_tag_entries(entry, config, "Exploit")

def count_issue_tracking_entries(entry, config):
    return count_tag_entries(entry, config, "Issue Tracking")

def count_mailing_list_entries(entry, config):
    return count_tag_entries(entry, config, "Mailing List")

def count_mitigation_entries(entry, config):
    return count_tag_entries(entry, config, "Mitigation")

def count_not_applicable_entries(entry, config):
    return count_tag_entries(entry, config, "Not Applicable")

def count_patch_entries(entry, config):
    return count_tag_entries(entry, config, "Patch")

def count_permissions_required_entries(entry, config):
    return count_tag_entries(entry, config, "Permissions Required")

def count_press_media_coverage_entries(entry, config):
    return count_tag_entries(entry, config, "Press/Media Coverage")

def count_product_entries(entry, config):
    return count_tag_entries(entry, config, "Product")

def count_release_notes_entries(entry, config):
    return count_tag_entries(entry, config, "Release Notes")

def count_technical_description_entries(entry, config):
    return count_tag_entries(entry, config, "Technical Description")

def count_third_party_advisory_entries(entry, config):
    return count_tag_entries(entry, config, "Third Party Advisory")

def count_url_repurposed_entries(entry, config):
    return count_tag_entries(entry, config, "URL Repurposed")

def count_us_government_resource_entries(entry, config):
    return count_tag_entries(entry, config, "US Government Resource")

def count_vdb_entry_entries(entry, config):
    return count_tag_entries(entry, config, "VDB Entry")

def count_vendor_advisory_entries(entry, config):
    return count_tag_entries(entry, config, "Vendor Advisory")



FEATURE2FUNCTION = {
    "cve": get_cve_id,
    "age": get_age,
    "cwe_id": get_cwe_id,
    "cvss_elements_exploitability_score": get_cvss_exploitability_score,
    "cvss_elements_impact_score": get_cvss_impact_score,
    "n_references": get_n_references,
    "description": get_description,
    "cve_vector": get_cve_vector,
    "exploit": count_exploit_entries,
    "issue_tracking": count_issue_tracking_entries,
    "mailing_list": count_mailing_list_entries,
    "mitigation": count_mitigation_entries,
    "not_applicable": count_not_applicable_entries,
    "patch": count_patch_entries,
    "permissions_required": count_permissions_required_entries,
    "press_media_coverage": count_press_media_coverage_entries,
    "product": count_product_entries,
    "release_notes": count_release_notes_entries,
    "technical_description": count_technical_description_entries,
    "third_party_advisory": count_third_party_advisory_entries,
    "url_repurposed": count_url_repurposed_entries,
    "us_government_resource": count_us_government_resource_entries,
    "vdb_entry": count_vdb_entry_entries,
    "vendor_advisory": count_vendor_advisory_entries
}
