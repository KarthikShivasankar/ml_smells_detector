from pathlib import Path
from typing import Dict, List
import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    # If nvidia gpu available
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    # E.g. M-series GPU
    DEVICE = torch.device("mps")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training_dir" / "cvss"
MODELS_PATH = Path(__file__).parent.parent.parent / "models"
LABEL_ENCODERS_PATH = MODELS_PATH / "cvss_label_encoders.json"

NUM_EPOCHS = 10


def save_label_encoders(
    label_encoders: Dict[str, Dict[str, int]], path: Path = LABEL_ENCODERS_PATH
) -> None:
    with open(path, "w") as f:
        json.dump({k: dict(v) for k, v in label_encoders.items()}, f)


def load_label_encoders(path: Path = LABEL_ENCODERS_PATH) -> Dict[str, Dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Label encoders file not found at {path}. Please train the model first."
        )
    with open(path, "r") as f:
        return json.load(f)


class TextMultiOutputDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        metadata: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
        label_encoders: Dict[str, Dict[str, int]] = None,
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
                if column != "filename" and column != "cve_id"
            }
        else:
            self.label_encoders = label_encoders

        self.num_classes_list = [
            len(encoder) for encoder in self.label_encoders.values()
        ]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        file_path = self.directory / row["filename"]
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
            if column != "filename" and column != "cve_id"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_label_encoders(self) -> Dict[str, Dict[str, int]]:
        return self.label_encoders


def initialize_dataloaders(tokenizer):
    metadata_file = "cvss_metadata.csv"
    test_size = 0.2
    val_size = 0.1  # 10% of the remaining data after test split

    full_metadata = pd.read_csv(os.path.join(DATA_DIR, metadata_file))
    label_encoders = {
        column: {label: idx for idx, label in enumerate(full_metadata[column].unique())}
        for column in full_metadata.columns
        if column != "filename" and column != "cve_id"
    }

    # First split: separate test set
    train_val_metadata, test_metadata = train_test_split(
        full_metadata, test_size=test_size, random_state=42
    )

    # Second split: separate validation set from remaining data
    train_metadata, val_metadata = train_test_split(
        train_val_metadata, test_size=val_size / (1 - test_size), random_state=42
    )

    train_dataset = TextMultiOutputDataset(
        DATA_DIR, train_metadata, tokenizer, label_encoders=label_encoders
    )
    val_dataset = TextMultiOutputDataset(
        DATA_DIR, val_metadata, tokenizer, label_encoders=label_encoders
    )
    test_dataset = TextMultiOutputDataset(
        DATA_DIR, test_metadata, tokenizer, label_encoders=label_encoders
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoders


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


def train_model(model, train_loader, val_loader, patience=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_PATH / "cvss_best_model.pth"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = tuple(label.to(DEVICE) for label in batch["labels"])

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs["loss"], outputs["logits"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if not batch_idx % 250:
                print(
                    f"Epoch: {epoch+1:04d}/{NUM_EPOCHS:04d} | "
                    f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                    f"Loss: {loss:.4f}"
                )

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = tuple(label.to(DEVICE) for label in batch["labels"])

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save best model so far
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model


def evaluate(model, loader, label_encoder, device):
    model.eval()
    with torch.set_grad_enabled(False):
        training_results = score_multioutput_model(model, loader, label_encoder, device)
        print("Test performance:")
        for classifier_id, metrics in training_results.items():
            print(f"{classifier_id} - ", end="\n")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.2f}%")
            print("\n")


def score_multioutput_model(model, data_loader, label_encoder, device):
    classifier_names = list(label_encoder.keys())
    all_predictions = [[] for _ in classifier_names]
    all_labels = [[] for _ in classifier_names]

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = [label.to(device) for label in batch["labels"]]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

            for idx, logit in enumerate(logits):
                predictions = (
                    torch.argmax(torch.softmax(logit, dim=-1), dim=-1).cpu().numpy()
                )
                all_predictions[idx].extend(predictions)
                all_labels[idx].extend(labels[idx].cpu().numpy())

    results = {}
    for i, (preds, lbls) in enumerate(zip(all_predictions, all_labels)):
        accuracy = accuracy_score(lbls, preds)
        precision = precision_score(lbls, preds, average="macro")
        recall = recall_score(lbls, preds, average="macro")
        f1 = f1_score(lbls, preds, average="macro")

        results[classifier_names[i]] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }

    return results


def initialize_model(
    num_labels: List[int], model_name: str = "distilbert-base-uncased"
):
    config = DistilBertConfig.from_pretrained(model_name, local_files_only=True)
    model = MultiOutputDistilBert(config, num_labels_list=num_labels)
    model.to(DEVICE)
    return model


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


def train_cvss_model():
    print(f"Using device: {DEVICE}")
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased",
        clean_up_tokenization_spaces=True,
        local_files_only=True,
    )
    train_loader, val_loader, test_loader, label_encoders = initialize_dataloaders(
        tokenizer
    )
    num_labels = [len(encoder) for encoder in label_encoders.values()]
    model = initialize_model(num_labels)
    _ = train_model(model, train_loader, val_loader)
    save_label_encoders(label_encoders)
    model_save_path = MODELS_PATH / "cvss_best_model.pth"
    model = load_multioutput_model(model_save_path)
    evaluate(model, test_loader, label_encoders, DEVICE)


if __name__ == "__main__":
    train_cvss_model()
