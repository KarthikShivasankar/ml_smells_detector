"""Tests for PyTorch-specific smell detection in FrameworkSpecificSmellDetector."""

from ml_code_smell_detector.detectors.framework_detector import FrameworkSpecificSmellDetector


def detect(tmp_py, code):
    detector = FrameworkSpecificSmellDetector()
    path = tmp_py(code)
    detector.detect_smells(path)
    return detector.smells


def names(smells):
    return [s["name"] for s in smells]


# ---------------------------------------------------------------------------
# Randomness Control Checker
# ---------------------------------------------------------------------------

class TestPyTorchRandomnessControl:
    def test_detects_torch_rand_without_seed(self, tmp_py):
        code = """\
            import torch
            x = torch.rand(10)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" in names(smells)

    def test_no_smell_with_manual_seed(self, tmp_py):
        code = """\
            import torch
            torch.manual_seed(42)
            x = torch.rand(10)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" not in names(smells)

    def test_detects_torch_randn_without_seed(self, tmp_py):
        code = """\
            import torch
            noise = torch.randn(3, 3)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker" in names(smells)


# ---------------------------------------------------------------------------
# Deterministic Algorithm Usage Checker
# ---------------------------------------------------------------------------

class TestDeterministicAlgorithmChecker:
    def test_detects_dataloader_without_deterministic(self, tmp_py):
        code = """\
            import torch
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset=[], batch_size=32)
        """
        smells = detect(tmp_py, code)
        assert "Deterministic Algorithm Usage Checker" in names(smells)

    def test_no_smell_with_deterministic_setting(self, tmp_py):
        code = """\
            import torch
            from torch.utils.data import DataLoader
            torch.use_deterministic_algorithms(True)
            loader = DataLoader(dataset=[], batch_size=32)
        """
        smells = detect(tmp_py, code)
        assert "Deterministic Algorithm Usage Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Randomness Control Checker (DataLoader with shuffle)
# ---------------------------------------------------------------------------

class TestDataLoaderRandomnessControl:
    def test_detects_shuffled_dataloader_without_worker_init(self, tmp_py):
        code = """\
            import torch
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset=[], batch_size=32, shuffle=True)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker (PyTorch-Dataloader)" in names(smells)

    def test_no_smell_with_worker_init_fn(self, tmp_py):
        code = """\
            import torch
            from torch.utils.data import DataLoader
            def seed_worker(worker_id):
                import numpy as np
                np.random.seed(42)
            loader = DataLoader(dataset=[], batch_size=32, shuffle=True, worker_init_fn=seed_worker)
        """
        smells = detect(tmp_py, code)
        assert "Randomness Control Checker (PyTorch-Dataloader)" not in names(smells)


# ---------------------------------------------------------------------------
# Gradient Clear Checker
# ---------------------------------------------------------------------------

class TestGradientClearChecker:
    def test_detects_missing_zero_grad(self, tmp_py):
        code = """\
            import torch
            import torch.optim as optim
            model = torch.nn.Linear(10, 1)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            loss = model(torch.rand(5, 10)).sum()
            loss.backward()
            optimizer.step()
        """
        smells = detect(tmp_py, code)
        assert "Gradient Clear Checker" in names(smells)

    def test_no_smell_with_zero_grad(self, tmp_py):
        code = """\
            import torch
            import torch.optim as optim
            model = torch.nn.Linear(10, 1)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = model(torch.rand(5, 10)).sum()
            loss.backward()
            optimizer.step()
        """
        smells = detect(tmp_py, code)
        assert "Gradient Clear Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Batch Normalisation Checker
# ---------------------------------------------------------------------------

class TestBatchNormalisationChecker:
    def test_detects_conv_without_batchnorm(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3),
            )
        """
        smells = detect(tmp_py, code)
        assert "Batch Normalisation Checker" in names(smells)

    def test_no_smell_with_batchnorm(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        """
        smells = detect(tmp_py, code)
        assert "Batch Normalisation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Dropout Usage Checker
# ---------------------------------------------------------------------------

class TestDropoutUsageChecker:
    def test_detects_deep_model_without_dropout(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            import torch.optim as optim
            model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            model.train()
            x = torch.rand(5, 100)
            loss = model(x).sum()
            loss.backward()
        """
        smells = detect(tmp_py, code)
        assert "Dropout Usage Checker" in names(smells)

    def test_no_smell_with_dropout(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(100, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.Dropout(0.3),
                nn.Linear(128, 10),
            )
            model.train()
            loss = torch.tensor(1.0)
            loss.backward()
        """
        smells = detect(tmp_py, code)
        assert "Dropout Usage Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Data Augmentation Checker
# ---------------------------------------------------------------------------

class TestDataAugmentationChecker:
    def test_detects_vision_task_without_augmentation(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder('img_dir/')
            loader = DataLoader(dataset, batch_size=32)
            model = nn.Conv2d(3, 64, 3)
            model.train()
            x = torch.rand(1, 3, 32, 32)
            out = model(x)
        """
        smells = detect(tmp_py, code)
        assert "Data Augmentation Checker" in names(smells)

    def test_no_smell_with_transforms(self, tmp_py):
        code = """\
            import torch
            from torch.utils.data import DataLoader
            from torchvision import transforms
            from torchvision.datasets import ImageFolder
            transform = transforms.Compose([transforms.RandomHorizontalFlip()])
            dataset = ImageFolder('data/', transform=transform)
            loader = DataLoader(dataset, batch_size=32)
            model = torch.nn.Linear(10, 1)
            model.train()
        """
        smells = detect(tmp_py, code)
        assert "Data Augmentation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Model Evaluation Checker
# ---------------------------------------------------------------------------

class TestModelEvaluationChecker:
    def test_detects_validation_without_eval_mode(self, tmp_py):
        code = """\
            import torch
            model = torch.nn.Linear(10, 1)
            val_loader = [torch.rand(5, 10)]
            for x in val_loader:
                output = model(x)
        """
        smells = detect(tmp_py, code)
        assert "Model Evaluation Checker" in names(smells)

    def test_no_smell_with_eval(self, tmp_py):
        code = """\
            import torch
            model = torch.nn.Linear(10, 1)
            model.eval()
            val_loader = [torch.rand(5, 10)]
            for x in val_loader:
                output = model(x)
        """
        smells = detect(tmp_py, code)
        # model.eval() is called: the 'eval' string appears in call.func.as_string()
        # so has_eval_mode is True and the smell should not be present.
        assert "Model Evaluation Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Logging Checker
# ---------------------------------------------------------------------------

class TestLoggingChecker:
    def test_detects_training_without_logging(self, tmp_py):
        code = """\
            import torch
            import torch.nn as nn
            model = nn.Linear(10, 1)
            model.train()
            x = torch.rand(5, 10)
            loss = model(x).sum()
            loss.backward()
        """
        smells = detect(tmp_py, code)
        assert "Logging Checker" in names(smells)

    def test_no_smell_with_tensorboard(self, tmp_py):
        code = """\
            import torch
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            model = torch.nn.Linear(10, 1)
            model.train()
            loss = torch.tensor(0.5)
            loss.backward()
            writer.add_scalar('Loss/train', loss.item(), 0)
        """
        smells = detect(tmp_py, code)
        assert "Logging Checker" not in names(smells)


# ---------------------------------------------------------------------------
# Framework label
# ---------------------------------------------------------------------------

class TestPyTorchFrameworkLabel:
    def test_pytorch_smells_have_correct_framework(self, tmp_py):
        code = """\
            import torch
            x = torch.rand(10)
        """
        smells = detect(tmp_py, code)
        pytorch_smells = [s for s in smells if s["framework"] == "PyTorch"]
        assert len(pytorch_smells) > 0
