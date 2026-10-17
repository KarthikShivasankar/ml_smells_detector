"""Fine-tune a sentiment model with the Hugging Face Trainer."""
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

checkpoint = "distilbert-base-uncased"
dataset = load_dataset("imdb", revision="main")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, force_download=True)  # expect: ML.HF.TOKENIZER_CACHING_NOT_USED
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # expect: ML.HF.MODEL_VERSIONING_NOT_SPECIFIED
local = AutoModelForSequenceClassification.from_pretrained("./checkpoints/best")


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)


encoded = dataset.map(tokenize, batched=True)
args = TrainingArguments(output_dir="out", num_train_epochs=3, per_device_train_batch_size=16)  # expect: ML.HF.DISTRIBUTED_TRAINING_NOT_CONFIGURED, ML.HF.GRADIENT_ACCUMULATION_NOT_CONFIGURED, ML.HF.MIXED_PRECISION_TRAINING_NOT_ENABLED
trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["test"])  # expect: ML.HF.EARLY_STOPPING_NOT_IMPLEMENTED
trainer.train()
