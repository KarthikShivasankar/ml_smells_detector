"""Tests for Hugging Face-specific smell detection in HuggingFaceSmellDetector."""

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

    def test_no_smell_with_revision_kwarg(self, tmp_py):
        # Regression for #19: revision= is how transformers pins a model version.
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased', revision='v1.0')
        """
        smells = detect(tmp_py, code)
        assert "Model versioning not specified" not in smell_texts(smells)

    def test_at_suffix_is_not_a_version_pin(self, tmp_py):
        # 'name@rev' is not supported by from_pretrained, so it must not clear the finding.
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased@v1.0')
        """
        smells = detect(tmp_py, code)
        assert "Model versioning not specified" in smell_texts(smells)

    def test_detects_concrete_model_classes_and_pipeline(self, tmp_py):
        code = """\
            from transformers import BertForMaskedLM, GPT2LMHeadModel, pipeline
            a = BertForMaskedLM.from_pretrained('bert-base-uncased')
            b = GPT2LMHeadModel.from_pretrained('gpt2')
            c = pipeline('text-generation', model='gpt2')
        """
        smells = detect(tmp_py, code)
        assert [s["line_number"] for s in smells if s["smell"] == "Model versioning not specified"] == [2, 3, 4]

    def test_no_smell_for_local_checkpoints_configs_and_tokenizers(self, tmp_py):
        code = """\
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            model = AutoModel.from_pretrained('./checkpoints/best')
            config = AutoConfig.from_pretrained('bert-base-uncased')
            tok = AutoTokenizer.from_pretrained('bert-base-uncased', max_length=128)
        """
        smells = detect(tmp_py, code)
        assert "Model versioning not specified" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Tokenizer Caching
# ---------------------------------------------------------------------------

class TestTokenizerCaching:
    def test_no_smell_with_default_cache(self, tmp_py):
        # Regression for #20: transformers caches downloads by default; cache_dir only moves the cache.
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Tokenizer caching not used" not in smell_texts(smells)

    def test_detects_force_download(self, tmp_py):
        code = """\
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', force_download=True)
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
    def test_no_smell_with_default_cache(self, tmp_py):
        # Regression for #20: transformers caches downloads by default; cache_dir only moves the cache.
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased')
        """
        smells = detect(tmp_py, code)
        assert "Model caching not used" not in smell_texts(smells)

    def test_detects_force_download(self, tmp_py):
        code = """\
            from transformers import AutoModel
            model = AutoModel.from_pretrained('bert-base-uncased', force_download=True)
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

    def test_no_smell_when_settings_passed_at_call_time(self, tmp_py):
        # Regression for #21: truncation/padding/max_length belong on the tokenizer call, not from_pretrained.
        code = """\
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained('bert-base-uncased', revision='v1.0')
            enc = tok('hello', truncation=True, padding='max_length', max_length=16, return_tensors='pt')
        """
        smells = detect(tmp_py, code)
        assert "Deterministic tokenization settings not specified" not in smell_texts(smells)

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
            from transformers import AutoModel, Trainer
            model = AutoModel.from_pretrained('bert-base-uncased')
            data = [{'text': 'hello'}]
            Trainer(model=model, train_dataset=data).train()
        """
        smells = detect(tmp_py, code)
        assert "Efficient data loading not detected" in smell_texts(smells)

    def test_no_smell_for_inference_only_script(self, tmp_py):
        code = """\
            from transformers import pipeline
            classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased', revision='main')
            print(classifier('hello'))
        """
        assert detect(tmp_py, code) == []

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
            args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
            )
            trainer = Trainer(args=args)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Mixed precision training not enabled" in smell_texts(smells)

    def test_no_smell_with_fp16(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                fp16=True,
            )
            trainer = Trainer(args=args)
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
            args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
            )
            trainer = Trainer(args=args)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Gradient accumulation not configured" in smell_texts(smells)

    def test_no_smell_with_gradient_accumulation_steps(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            args = TrainingArguments(
                output_dir='./results',
                gradient_accumulation_steps=4,
            )
            trainer = Trainer(args=args)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Gradient accumulation not configured" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

class TestLearningRateScheduling:
    def test_detects_custom_loop_without_scheduler(self, tmp_py):
        code = """\
            from transformers import AutoModelForSequenceClassification
            from torch.optim import AdamW
            model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', revision='v1')
            optimizer = AdamW(model.parameters(), lr=5e-5)
            for batch in []:
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
        """
        smells = detect(tmp_py, code)
        assert [s["line_number"] for s in smells if s["smell"] == "Learning rate scheduler not detected"] == [7]

    def test_no_smell_with_get_scheduler(self, tmp_py):
        code = """\
            from transformers import get_scheduler
            from torch.optim import AdamW
            optimizer = AdamW([], lr=5e-5)
            scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=10)
            for batch in []:
                batch.loss.backward()
                optimizer.step()
                scheduler.step()
        """
        smells = detect(tmp_py, code)
        assert "Learning rate scheduler not detected" not in smell_texts(smells)

    def test_no_smell_for_trainer_default_linear_schedule(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            args = TrainingArguments(output_dir='./results', num_train_epochs=3)
            trainer = Trainer(args=args)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Learning rate scheduler not detected" not in smell_texts(smells)


class TestTrainingArgumentsChecks:
    def test_reported_at_the_training_arguments_call(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            args = TrainingArguments(output_dir='./results')
            Trainer(args=args).train()
        """
        smells = detect(tmp_py, code)
        lines = {s["smell"]: s["line_number"] for s in smells}
        for smell in ("Mixed precision training not enabled", "Gradient accumulation not configured",
                      "Distributed training not configured"):
            assert lines[smell] == 2

    def test_non_literal_values_do_not_abort_detection(self, tmp_py, capsys):
        code = """\
            from transformers import Trainer, TrainingArguments
            USE_AMP = True
            STEPS = 4
            args = TrainingArguments(output_dir='o', fp16=USE_AMP, gradient_accumulation_steps=STEPS, fsdp='full_shard')
            Trainer(args=args).train()
        """
        smells = detect(tmp_py, code)
        assert capsys.readouterr().err == ""
        assert not {"Mixed precision training not enabled", "Gradient accumulation not configured",
                    "Distributed training not configured"} & set(smell_texts(smells))

    def test_kwargs_unpacking_is_not_reported(self, tmp_py):
        code = """\
            from transformers import TrainingArguments
            config = {'fp16': True}
            args = TrainingArguments(output_dir='o', **config)
        """
        smells = detect(tmp_py, code)
        assert "Mixed precision training not enabled" not in smell_texts(smells)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_detects_training_without_early_stopping(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments
            args = TrainingArguments(output_dir='./results')
            trainer = Trainer(args=args)
            trainer.train()
        """
        smells = detect(tmp_py, code)
        assert "Early stopping not implemented" in smell_texts(smells)

    def test_no_smell_with_early_stopping_callback(self, tmp_py):
        code = """\
            from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
            args = TrainingArguments(output_dir='./results')
            trainer = Trainer(
                args=args,
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
