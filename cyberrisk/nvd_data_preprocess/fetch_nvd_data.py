import requests
import zipfile
import os
import json
from pathlib import Path
from typing import List

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}


def download_zipfile(year, nvd_filepath):
    url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.zip"
    try:
        response = requests.get(url, stream=True, headers=HEADERS)
        response.raise_for_status()
        filepath = nvd_filepath / f"nvdcve-1.1-{year}.json.zip"

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File for {year} downloaded successfully.")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file for {year}: {e}")
        return None


def extract_zipfile(zip_filepath, nvd_filepath) -> None:
    try:
        with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(nvd_filepath)
        os.remove(zip_filepath)
        json_filepath = nvd_filepath / f"nvdcve-1.1-{zip_filepath.stem.split('-')[-1]}"
        print(f"Data extracted successfully: {json_filepath}")
    except Exception as e:
        print(f"Error extracting the file: {e}")


def download_and_process_cve_data(nvd_filepath: Path):
    nvd_filepath.mkdir(parents=True, exist_ok=True)

    for year in range(2002, 2025):
        zip_filepath = download_zipfile(year, nvd_filepath)
        if zip_filepath:
            extract_zipfile(zip_filepath, nvd_filepath)


if __name__ == "__main__":
    NVD_FILEPATH = Path(__file__).parent.parent.resolve() / "cve_corpus"
    download_and_process_cve_data(NVD_FILEPATH)
