import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
from cvss import CVSS3
from transformers import DistilBertTokenizer, DistilBertConfig

from cwe_cvss_prediction.cvss_classifier import MultiOutputDistilBert
from cwe_cvss_prediction.cwe_classifier import DistilBertForMultilabelClassification

DATA_PATH = Path(__file__).parent.parent / "data"
MODELS_PATH = Path(__file__).parent.parent / "models"

CWE_LABEL_ENCODERS_PATH = MODELS_PATH / "cwe_label_encoders.json"
CVSS_LABEL_ENCODERS_PATH = MODELS_PATH / "cvss_label_encoders.json"
CWE_BEST_PATH = MODELS_PATH / "cwe_best_model.pth"
CVSS_BEST_PATH = MODELS_PATH / "cvss_best_model.pth"
INPUT_PATH = DATA_PATH / "cve_corpus"
OUTPUT_FOLDER = DATA_PATH / "nvd_data_with_predictions"


CVSS_METRIC_ORDER: List[str] = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
METRIC_MAPPINGS = {
    "AV": {"N": "NETWORK", "A": "ADJACENT_NETWORK", "L": "LOCAL", "P": "PHYSICAL"},
    "AC": {"L": "LOW", "H": "HIGH"},
    "PR": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "UI": {"N": "NONE", "R": "REQUIRED"},
    "S": {"U": "UNCHANGED", "C": "CHANGED"},
    "C": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "I": {"N": "NONE", "L": "LOW", "H": "HIGH"},
    "A": {"N": "NONE", "L": "LOW", "H": "HIGH"},
}


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    # If nvidia gpu available
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    # E.g. M-series GPU
    DEVICE = torch.device("mps")


def load_json_data(json_filepath):
    try:
        with open(json_filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


def load_multioutput_model(
    model_path,
    num_labels: List[int] = [4, 2, 3, 2, 2, 3, 3, 3],
    model_name: str = "distilbert-base-uncased",
):
    config = DistilBertConfig.from_pretrained(model_name, local_files_only=True)
    model = MultiOutputDistilBert(config, num_labels_list=num_labels)
    state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_multilabel_model(
    model_path,
    num_labels: int = 422,
    model_name: str = "distilbert-base-uncased",
):
    config = DistilBertConfig.from_pretrained(
        model_name, num_labels=num_labels, local_files_only=True
    )
    model = DistilBertForMultilabelClassification(config)
    state_dict = torch.load(model_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_cvss_vector(model, tokenizer, text: str, label_encoders):
    model.eval()
    model.to(DEVICE)

    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

    # Convert logits to actual labels using the label encoders
    cvss_metrics = {}
    for idx, logit in enumerate(logits):
        predicted_idx = torch.argmax(logit, dim=-1).item()
        classifier_name = list(label_encoders.keys())[idx]
        label_decoder = {v: k for k, v in label_encoders[classifier_name].items()}
        cvss_metrics[classifier_name] = label_decoder[predicted_idx]
    cvss_vector_parts = [f"{key}:{cvss_metrics[key]}" for key in CVSS_METRIC_ORDER]
    cvss_vector = "CVSS:3.1/" + "/".join(cvss_vector_parts)
    return cvss_vector


def predict_cwes(model, tokenizer, text: str, label_encoder, max_labels=3) -> List[str]:
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    idx_to_label = {idx: label for label, idx in label_encoder.items()}
    top_indices = np.argsort(probabilities)[::-1]

    predicted_labels = [
        idx_to_label[idx] for idx in top_indices if probabilities[idx] > 0.5
    ]

    # If no predictions are above 0.5, take the single highest probability prediction
    if not predicted_labels:
        top_index = top_indices[0]
        predicted_labels = [idx_to_label[top_index]]

    return predicted_labels[:max_labels]


def get_full_metric_value(metric: str, value: str) -> str:
    return METRIC_MAPPINGS.get(metric, {}).get(value, value)


def extract_description(cve_item: Dict) -> str:
    description_data = (
        cve_item.get("cve", {}).get("description", {}).get("description_data", [])
    )
    if description_data and isinstance(description_data[0], dict):
        return description_data[0].get("value", "")
    return ""


def calculate_cvss_metrics(predicted_vector: str) -> Dict[str, Union[str, float]]:
    cvss_calculator = CVSS3(predicted_vector)
    cvss_calculator.compute_isc()
    cvss_calculator.compute_esc()

    return {
        "version": "3.1",
        "vectorString": predicted_vector,
        "attackVector": get_full_metric_value("AV", cvss_calculator.metrics["AV"]),
        "attackComplexity": get_full_metric_value("AC", cvss_calculator.metrics["AC"]),
        "privilegesRequired": get_full_metric_value(
            "PR", cvss_calculator.metrics["PR"]
        ),
        "userInteraction": get_full_metric_value("UI", cvss_calculator.metrics["UI"]),
        "scope": get_full_metric_value("S", cvss_calculator.metrics["S"]),
        "confidentialityImpact": get_full_metric_value(
            "C", cvss_calculator.metrics["C"]
        ),
        "integrityImpact": get_full_metric_value("I", cvss_calculator.metrics["I"]),
        "availabilityImpact": get_full_metric_value("A", cvss_calculator.metrics["A"]),
        "baseScore": float(cvss_calculator.base_score),
        "baseSeverity": cvss_calculator.severities()[0].upper(),
        "exploitabilityScore": float(cvss_calculator.esc),
        "impactScore": float(cvss_calculator.isc),
    }


def process_nvd_json(
    input_file: Path,
    cvss_model,
    cwe_model,
    tokenizer,
    cvss_label_encoders,
    cwe_label_encoder,
) -> None:
    print(f"Processing: {input_file}")

    nvd_data = load_json_data(input_file)

    for cve_item in nvd_data.get("CVE_Items", []):
        description = extract_description(cve_item)

        predicted_vector = predict_cvss_vector(
            cvss_model, tokenizer, description, cvss_label_encoders
        )
        predicted_cwes = predict_cwes(
            cwe_model, tokenizer, description, cwe_label_encoder
        )

        # Add predicted CWEs to the problemtype field
        if "cve" not in cve_item:
            cve_item["cve"] = {}
        if "problemtype" not in cve_item["cve"]:
            cve_item["cve"]["problemtype"] = {"problemtype_data": []}

        # Add predicted CWEs
        predicted_problemtype = {
            "description": [{"lang": "en", "value": cwe} for cwe in predicted_cwes]
        }

        cve_item["cve"]["problemtype"]["predicted_problemtype_data"] = [
            predicted_problemtype
        ]

        # Add predicted CVSS
        if "impact" not in cve_item:
            cve_item["impact"] = {}

        cvss_metrics = calculate_cvss_metrics(predicted_vector)

        cve_item["impact"]["predicted_baseMetricV3"] = cvss_metrics

    output_filepath = OUTPUT_FOLDER / input_file.name
    with open(output_filepath, "w") as f:
        json.dump(nvd_data, f, indent=2)


def process_multiple_nvd_jsons(
    input_path: Path,
    cvss_model,
    cwe_model,
    tokenizer,
    cvss_label_encoder,
    cwe_label_encoder,
) -> None:
    # Create output folder if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Process each JSON file in the input folder
    for json_file in input_path.glob("*.json"):
        try:
            process_nvd_json(
                json_file,
                cvss_model,
                cwe_model,
                tokenizer,
                cvss_label_encoder,
                cwe_label_encoder,
            )
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")


def predict_cwe_and_cvss_for_nvd_data(model_name: str = "distilbert-base-uncased"):
    print(f"Using device: {DEVICE}")
    cvss_label_encoder = load_json_data(CVSS_LABEL_ENCODERS_PATH)
    cwe_label_encoder = load_json_data(CWE_LABEL_ENCODERS_PATH)

    tokenizer = DistilBertTokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=True
    )
    cvss_model = load_multioutput_model(CVSS_BEST_PATH, model_name=model_name)
    cwe_model = load_multilabel_model(
        CWE_BEST_PATH, model_name=model_name, num_labels=len(cwe_label_encoder)
    )

    process_multiple_nvd_jsons(
        INPUT_PATH,
        cvss_model,
        cwe_model,
        tokenizer,
        cvss_label_encoder,
        cwe_label_encoder,
    )


if __name__ == "__main__":
    predict_cwe_and_cvss_for_nvd_data()
