"""Tests for Hugging Face-specific smell detection in HuggingFaceSmellDetector."""

import pytest
from ml_code_smell_detector.detectors.huggingface_detector import HuggingFaceSmellDetector


def detect(tmp_py, code):
    detector = HuggingFaceSmellDetector()
    path = tmp_py(code)
    detector.detect_smells(path)
    return detector.smells


def smell_texts(smells):
    return [s["smell"] for s in smells]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestHuggingFaceDetectorInit:
    def test_starts_with_empty_smells(self):
        detector = HuggingFaceSmellDetector()
        assert detector.smells == []

    def test_skips_non_transformers_file(self, tmp_py):
        smells = detect(tmp_py, """\
            import numpy as np
            x = np.array([1, 2, 3])
        """)
        assert smells == []


# ---------------------------------------------------------------------------
# Model Versioning
# ---------------------------------------------------------------------------

class TestModelVersioning:
    def test_detects_missing_revision(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Model versioning not specified" in smell_texts(smells)

    def test_no_smell_with_revision_tag(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased@v1.0')
        """
        smells = detect(tmp_py, code)
        assert "Model versioning not specified" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Tokenizer Caching
# ---------------------------------------------------------------------------

class TestTokenizerCaching:
    def test_detects_missing_cache_dir(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Tokenizer caching not used" in smell_texts(smells)

    def test_no_smell_with_cache_dir(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='/tmp/hf')
        """
        smells = detect(tmp_py, code)
        assert "Tokenizer caching not used" not in smell_texts(smells)

    def test_no_smell_with_local_files_only(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        """
        smells = detect(tmp_py, code)
        assert "Tokenizer caching not used" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Model Caching
# ---------------------------------------------------------------------------

class TestModelCaching:
    def test_detects_missing_cache_dir(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Model caching not used" in smell_texts(smells)

    def test_no_smell_with_cache_dir(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased', cache_dir='/tmp/hf')
        """
        smells = detect(tmp_py, code)
        assert "Model caching not used" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Deterministic Tokenization
# ---------------------------------------------------------------------------

class TestDeterministicTokenization:
    def test_detects_missing_tokenization_params(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Deterministic tokenization settings not specified" in smell_texts(smells)

    def test_no_smell_with_padding_and_truncation(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                'bert-base-uncased',
                padding=True,
                truncation=True,
                max_length=512
            )
        """
        smells = detect(tmp_py, code)
        assert "Deterministic tokenization settings not specified" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Efficient Data Loading
# ---------------------------------------------------------------------------

class TestEfficientDataLoading:
    def test_detects_missing_efficient_loading(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased')
            data = [{'text': 'hello'}]
        """
        smells = detect(tmp_py, code)
        assert "Efficient data loading not detected" in smell_texts(smells)

    def test_no_smell_with_load_dataset(self, tmp_py):
        code = """\
            from transformers import AutoModel
            from datasets import load_dataset
            model = AutoModel.from_pretrained('bert-base-uncased')
            dataset = load_dataset('imdb')
        """
        smells = detect(tmp_py, code)
        assert "Efficient data loading not detected" not in smell_texts(smells)

    def test_no_smell_with_dataloader(self, tmp_py):
        code = """\
            from transformers import AutoModel
            from torch.utils.data import DataLoader
            model = AutoModel.from_pretrained('bert-base-uncased')
            loader = DataLoader([])
        """
        smells = detect(tmp_py, code)
        assert "Efficient data loading not detected" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Mixed Precision Training
# ---------------------------------------------------------------------------

class TestMixedPrecisionTraining:
    def test_detects_training_without_fp16(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Mixed precision training not enabled" in smell_texts(smells)

    def test_no_smell_with_fp16(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                fp16=True,
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Mixed precision training not enabled" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Gradient Accumulation
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    def test_detects_training_without_grad_accum(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Gradient accumulation not configured" in smell_texts(smells)

    def test_no_smell_with_gradient_accumulation_steps(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                gradient_accumulation_steps=4,
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Gradient accumulation not configured" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

class TestLearningRateScheduling:
    def test_detects_training_without_lr_scheduler(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Learning rate scheduler not detected" in smell_texts(smells)

    def test_no_smell_with_lr_scheduler_type(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(
                output_dir='./results',
                lr_scheduler_type='cosine',
            )
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Learning rate scheduler not detected" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_detects_training_without_early_stopping(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            TrainingArguments = TrainingArguments(output_dir='./results')
            trainer = Trainer(args=TrainingArguments)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Early stopping not implemented" in smell_texts(smells)

    def test_no_smell_with_early_stopping_callback(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
            TrainingArguments = TrainingArguments(output_dir='./results')
            trainer = Trainer(
                args=TrainingArguments,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Early stopping not implemented" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Report and results
# ---------------------------------------------------------------------------

class TestHuggingFaceReport:
    def test_generate_report_format(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        """
        detector = HuggingFaceSmellDetector()
        path = tmp_py(code)
        detector.detect_smells(path)
        report = detector.generate_report()
        assert "Hugging Face Code Smell Report" in report
        assert "Total smells detected" in report

    def test_get_results_returns_correct_keys(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        """
        detector = HuggingFaceSmellDetector()
        path = tmp_py(code)
        detector.detect_smells(path)
        results = detector.get_results()
        assert isinstance(results, list)
        assert len(results) > 0
        r = results[0]
        assert r["framework"] == "Hugging Face"
        assert "name" in r
        assert "fix" in r
        assert "benefits" in r
