import argparse

from pathlib import Path

import pandas as pd

from fetch_nvd_data import download_and_process_cve_data
from training_set_extraction import extract_training_set
from feature_extraction import process_cve_entries
from cwe_cvss_prediction.cvss_classifier import train_cvss_model
from cwe_cvss_prediction.cwe_classifier import train_cwe_classifier
from predict_cvss_cwe_for_nvd import predict_cwe_and_cvss_for_nvd_data

DATA_DIR = Path(__file__).parent.parent / "data"

NVD_FILEPATH = DATA_DIR / "cve_corpus"
TRAINING_DIR = DATA_DIR / "training_dir"
NVD_ENRICHED = DATA_DIR / "nvd_data_with_predictions"
PREPROCESS_FILEPATH = DATA_DIR / "preprocessed"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "fetch_corpus",
            "prepare_data",
            "train_cvss_model",
            "train_cwe_model",
            "enrich_corpus",
            "preprocess",
            "feature_generation"
        ],
    )
    parser.add_argument("--input", help="Input text for prediction mode")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "fetch_corpus":
        download_and_process_cve_data(nvd_filepath=NVD_FILEPATH)
    elif args.mode == "prepare_data":
        extract_training_set(data_dir=TRAINING_DIR, nvd_filepath=NVD_FILEPATH)
    elif args.mode == "train_cvss_model":
        train_cvss_model()
    elif args.mode == "train_cwe_model":
        train_cwe_classifier()
    elif args.mode == "enrich_corpus":
        predict_cwe_and_cvss_for_nvd_data()
    elif args.mode == "preprocess":
        cve_data = process_cve_entries(nvd_filepath=NVD_ENRICHED)
        df = pd.DataFrame(cve_data)
        PREPROCESS_FILEPATH.mkdir(parents=True, exist_ok=True)
        save_filepath = (
            PREPROCESS_FILEPATH / "processed_nvd_data_with_predictions.csv"
        )
        df.to_csv(save_filepath, index=False)
        print(f"Data has been processed and saved to '{save_filepath}'")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")



if __name__ == "__main__":
    main()
