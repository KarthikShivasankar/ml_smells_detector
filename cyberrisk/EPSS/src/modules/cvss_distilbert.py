import json
import os
import sys

sys.path.append("..")

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from src.utils.evaluation import score_multioutput_model


from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

CVSS_METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
LABEL_ENCODERS = {
    "AV": {"N": 0, "L": 1, "A": 2, "P": 3},
    "AC": {"L": 0, "H": 1},
    "PR": {"N": 0, "L": 1, "H": 2},
    "UI": {"R": 0, "N": 1},
    "S": {"C": 0, "U": 1},
    "C": {"L": 0, "H": 1, "N": 2},
    "I": {"L": 0, "H": 1, "N": 2},
    "A": {"N": 0, "H": 1, "L": 2},
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NVD_FILEPATH = Path(__file__).parent.parent.parent / "NVD_jsons"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_PATH = MODELS_DIR / "cvss_distilbert.pth"
NUM_EPOCHS = 10


class DistilbertClassifierCVSS:
    def __init__(self) -> None:
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.model = MultiOutputDistilBert(
            config, num_labels_list=[4, 2, 3, 2, 2, 3, 3, 3]
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.should_train = True
        if MODEL_PATH.is_file():
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.should_train = False
        else:
            print("No model in models dir")

    def train(self, nvd_features: pd.DataFrame) -> None:
        if not self.should_train:
            print("Model is already trained and loaded. Skipping training.")
            return

        print(f"Creating new dataset based on files in {str(NVD_FILEPATH)}")
        data_list = []
        for year in [str(year) for year in range(2015, 2024)]:
            filepath = NVD_FILEPATH / f"nvdcve-1.1-{year}.json"
            if not filepath.is_file():
                continue
            with open(filepath, "r") as f:
                data = json.load(f)
                data_list.append(data)
        extract_training_set(data_list)

        train_loader, test_loader, label_encoders = self.initialize_dataloaders()
        optimizer = AdamW(self.model.parameters(), lr=3e-4)
        self.model.train()

        for epoch in range(NUM_EPOCHS):
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = tuple(label.to(DEVICE) for label in batch["labels"])

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss, logits = outputs["loss"], outputs["logits"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not batch_idx % 250:
                    print(
                        f"Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} | "
                        f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                        f"Loss: {loss:.4f}"
                    )

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        self.evaluate(test_loader, label_encoders)

    def predict(self, nvd_features: pd.DataFrame) -> pd.DataFrame:
        missing_indices = nvd_features.index[
            nvd_features["cve_vector"] == "NVD-CVE-vector-noinfo"
        ]
        for idx in missing_indices:
            description = nvd_features.loc[idx, "description"]
            cvss_metrics = self.predict_cvss(description, LABEL_ENCODERS)
            cvss_vector = "CVSS:3.1/" + "/".join(
                f"{key}:{cvss_metrics[key]}" for key in CVSS_METRIC_ORDER
            )
            nvd_features.loc[idx, "cve_vector"] = cvss_vector

        return nvd_features

    def predict_cvss(self, text: str, label_encoders: Dict) -> Dict:
        self.model.eval()
        self.model.to(DEVICE)

        tokens = self.tokenizer.encode_plus(
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        # Convert logits to actual labels using the label encoders
        cvss_vector = {}
        for idx, logit in enumerate(logits):
            predicted_idx = torch.argmax(logit, dim=-1).item()
            classifier_name = list(label_encoders.keys())[idx]
            label_decoder = {v: k for k, v in label_encoders[classifier_name].items()}
            cvss_vector[classifier_name] = label_decoder[predicted_idx]

        return cvss_vector

    def initialize_dataloaders(self):
        directory = Path.home() / "data" / "cve_data"
        metadata_file = "metadata.csv"
        test_size = 0.2
        full_metadata = pd.read_csv(os.path.join(directory, metadata_file))
        label_encoders = {
            column: {
                label: idx for idx, label in enumerate(full_metadata[column].unique())
            }
            for column in full_metadata.columns
            if column != "filename"
        }
        train_metadata, test_metadata = train_test_split(
            full_metadata, test_size=test_size, random_state=42
        )
        train_dataset = TextMultiOutputDataset(
            directory, train_metadata, self.tokenizer, label_encoders=label_encoders
        )
        test_dataset = TextMultiOutputDataset(
            directory, test_metadata, self.tokenizer, label_encoders=label_encoders
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader, label_encoders

    def evaluate(self, loader, label_encoder: Dict):
        self.model.eval()
        with torch.set_grad_enabled(False):
            training_results = score_multioutput_model(
                self.model, loader, label_encoder, DEVICE
            )
            print("Training performance:")
            for classifier_id, metrics in training_results.items():
                print(f"{classifier_id} - ", end="")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.2f}%")


def extract_training_set(cve_data_list: List):
    """
    Extracts the training set from corpus and places under home
    """
    base_path = Path.home() / "data" / "cve_data"
    base_path.mkdir(parents=True, exist_ok=True)

    data_entries = []

    for cve_data in cve_data_list:
        for entry in cve_data["CVE_Items"]:
            if "baseMetricV3" not in entry["impact"]:
                continue
            description = entry["cve"]["description"]["description_data"][0]["value"]

            vector_string = entry["impact"]["baseMetricV3"]["cvssV3"]["vectorString"]
            metrics = vector_string.split("/")[1:]

            metric_dict = {}
            for metric in metrics:
                metric_name, metric_value = metric.split(":")
                metric_dict[metric_name] = metric_value

            cve_id = entry["cve"]["CVE_data_meta"]["ID"]
            file_path = base_path / f"{cve_id}.txt"
            file_path.write_text(description)

            data_entries.append({"filename": file_path.name, **metric_dict})

    metadata_df = pd.DataFrame(data_entries)
    metadata_df.to_csv(base_path / "metadata.csv", index=False)


class MultiOutputDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, num_labels_list):
        super().__init__(config)
        self.pre_classifier = None
        self.classifier = None

        self.classifiers = nn.ModuleList(
            [nn.Linear(config.dim, num_labels) for num_labels in num_labels_list]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        logits = [classifier(pooled_output) for classifier in self.classifiers]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = sum(loss_fct(logit, label) for logit, label in zip(logits, labels))
            return {"loss": loss, "logits": logits} if return_dict else (loss, logits)

        return {"logits": logits} if return_dict else logits


class TextMultiOutputDataset(Dataset):
    def __init__(
        self, directory, metadata, tokenizer, max_length=512, label_encoders=None
    ):
        self.metadata = metadata
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        if label_encoders is None:
            self.label_encoders = {
                column: {
                    label: idx for idx, label in enumerate(metadata[column].unique())
                }
                for column in metadata.columns
                if column != "filename"
            }
        else:
            self.label_encoders = label_encoders

        self.num_classes_list = [
            len(encoder) for encoder in self.label_encoders.values()
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.directory, row["filename"])
        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        tokens = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        labels = tuple(
            torch.tensor(self.label_encoders[column][row[column]])
            for column in self.metadata.columns
            if column != "filename"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_label_encoders(self):
        return self.label_encoders
