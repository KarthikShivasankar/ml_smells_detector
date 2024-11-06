import os
import json
import torch
import random
from pathlib import Path
import pandas as pd
from typing import Dict
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

# Constants
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training_dir" / "cwe"
MODELS_PATH = Path(__file__).parent.parent.parent / "models"
LABEL_ENCODERS_PATH = MODELS_PATH / "cwe_label_encoders.json"


class DistilBertForMultilabelClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)
        else:
            return logits


class CWEDataset(Dataset):
    def __init__(
        self,
        directory,
        metadata,
        tokenizer,
        max_length=512,
        label_encoder=None,
        max_cwes=3,
    ):
        self.metadata = metadata
        self.directory = directory
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_cwes = max_cwes

        if label_encoder is None:
            all_cwes = set()
            for cwes_json in metadata["cwes"]:
                all_cwes.update(json.loads(cwes_json))
            self.label_encoder = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}
        else:
            self.label_encoder = label_encoder

        self.num_classes = len(self.label_encoder)

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

        cwes = json.loads(row["cwes"])
        if len(cwes) > self.max_cwes:
            cwes = random.sample(cwes, self.max_cwes)

        labels = torch.zeros(self.num_classes)
        for cwe in cwes:
            if cwe in self.label_encoder:
                labels[self.label_encoder[cwe]] = 1

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def get_label_encoder(self):
        return self.label_encoder


def initialize_dataloaders(tokenizer, data_dir, metadata_file="cwe_metadata.csv", batch_size=64):
    test_size = 0.2
    val_size = 0.1  # 10% of the remaining data after test split

    full_metadata = pd.read_csv(os.path.join(data_dir, metadata_file))
    
    all_cwes = set()
    for cwes_json in full_metadata['cwes']:
        all_cwes.update(json.loads(cwes_json))
    label_encoder = {cwe: idx for idx, cwe in enumerate(sorted(all_cwes))}

    train_val_metadata, test_metadata = train_test_split(
        full_metadata, test_size=test_size, random_state=42
    )

    train_metadata, val_metadata = train_test_split(
        train_val_metadata, test_size=val_size / (1 - test_size), random_state=42
    )

    train_dataset = CWEDataset(
        data_dir, train_metadata, tokenizer, label_encoder=label_encoder, max_cwes=3
    )
    val_dataset = CWEDataset(
        data_dir, val_metadata, tokenizer, label_encoder=label_encoder, max_cwes=3
    )
    test_dataset = CWEDataset(
        data_dir, test_metadata, tokenizer, label_encoder=label_encoder, max_cwes=3
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoder



def save_label_encoders(
    label_encoders: Dict[str, Dict[str, int]], path: Path = LABEL_ENCODERS_PATH
) -> None:
    with open(path, "w") as f:
        json.dump({k: v for k, v in label_encoders.items()}, f)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(
                input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="micro")
    return avg_loss, f1


def train_model(
    model, train_loader, val_loader, device, num_epochs=10, patience=3, lr=5e-5
):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_PATH / "cwe_best_model.pth"

    optimizer = AdamW(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return model


def train_cwe_classifier(
    data_dir=DATA_DIR,
    base_model: str = "distilbert-base-uncased",
    label_encoders_path=LABEL_ENCODERS_PATH,
    num_epochs=10,
    patience=3,
    lr=5e-5,
):
    print(f"Using device: {DEVICE}")

    tokenizer = DistilBertTokenizer.from_pretrained(
        base_model, clean_up_tokenization_spaces=True, local_files_only=True
    )
    train_loader, val_loader, test_loader, label_encoder = initialize_dataloaders(
        tokenizer, data_dir
    )
    num_labels = len(label_encoder)

    config = DistilBertConfig.from_pretrained(
        base_model, num_labels=num_labels, local_files_only=True
    )
    model = DistilBertForMultilabelClassification(config)
    model.to(DEVICE)

    save_label_encoders(label_encoder, path=label_encoders_path)

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        DEVICE,
        num_epochs=num_epochs,
        patience=patience,
        lr=lr,
    )

    # Evaluate the model on the test set
    test_loss, test_f1 = evaluate(trained_model, test_loader, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test F1 Score: {test_f1:.4f}")


if __name__ == "__main__":
    train_cwe_classifier()
