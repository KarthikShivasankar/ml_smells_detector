import astroid
from astroid import nodes
from typing import List, Dict, Any

class HuggingFaceSmellDetector:
    def __init__(self):
        self.smells: List[Dict[str, Any]] = []

    def detect_smells(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r') as file:
            content = file.read()
        module = astroid.parse(content)
        self.visit_module(module, file_path)
        return self.smells

    def visit_module(self, node: nodes.Module, file_path: str):
        self.check_model_versioning(node, file_path)
        self.check_tokenizer_caching(node, file_path)
        self.check_model_caching(node, file_path)
        self.check_deterministic_tokenization(node, file_path)
        self.check_efficient_data_loading(node, file_path)
        self.check_distributed_training(node, file_path)
        self.check_mixed_precision_training(node, file_path)
        self.check_gradient_accumulation(node, file_path)
        self.check_learning_rate_scheduling(node, file_path)
        self.check_early_stopping(node, file_path)

    def add_smell(self, smell: str, fix: str, benefits: str, strategies: str, node: nodes.NodeNG, file_path: str):
        self.smells.append({
            "smell": smell,
            "how_to_fix": fix,
            "benefits": benefits,
            "strategies": strategies,
            "line_number": node.lineno,
            "code_snippet": node.as_string(),
            "file_path": file_path
        })

    def check_model_versioning(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'from_pretrained' in call.func.as_string():
                if not any('@' in arg.as_string() for arg in call.args):
                    self.add_smell(
                        "Model versioning not specified",
                        "Specify model version when loading pre-trained models",
                        "Ensures consistency and reproducibility of results",
                        "Use model_name_or_path@revision when loading models (e.g., bert-base-uncased@v1)",
                        call,
                        file_path
                    )

    def check_tokenizer_caching(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'from_pretrained' in call.func.as_string() and 'Tokenizer' in call.func.as_string():
                if not any('cache_dir' in keyword.arg for keyword in call.keywords):
                    self.add_smell(
                        "Tokenizer caching not used",
                        "Cache tokenizers to avoid re-downloading",
                        "Reduces loading time and network dependency",
                        "Use cache_dir parameter when loading tokenizers",
                        call,
                        file_path
                    )

    def check_model_caching(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'from_pretrained' in call.func.as_string() and 'Model' in call.func.as_string():
                if not any('cache_dir' in keyword.arg for keyword in call.keywords):
                    self.add_smell(
                        "Model caching not used",
                        "Cache models to avoid re-downloading",
                        "Improves loading efficiency and reduces network dependency",
                        "Use cache_dir parameter when loading models",
                        call,
                        file_path
                    )

    def check_deterministic_tokenization(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'from_pretrained' in call.func.as_string() and 'Tokenizer' in call.func.as_string():
                if not any(keyword.arg in ['do_lower_case', 'strip_accents'] for keyword in call.keywords):
                    self.add_smell(
                        "Deterministic tokenization settings not specified",
                        "Use consistent tokenization settings",
                        "Ensures reproducible pre-processing and consistent model inputs",
                        "Set tokenization parameters explicitly when loading tokenizers",
                        call,
                        file_path
                    )

    def check_efficient_data_loading(self, node: nodes.Module, file_path: str):
        datasets_used = any('datasets' in import_node.names[0][0] for import_node in node.nodes_of_class(nodes.ImportFrom))
        if not datasets_used:
            self.add_smell(
                "Efficient data loading not detected",
                "Use efficient data loading techniques",
                "Enhances data processing speed and model training efficiency",
                "Use datasets library for loading and processing data",
                node,
                file_path
            )

    def check_distributed_training(self, node: nodes.Module, file_path: str):
        distributed_config = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg in ['distributed', 'tpu'] for keyword in assign.value.keywords):
                    distributed_config = True
                    break
        if not distributed_config:
            self.add_smell(
                "Distributed training not configured",
                "Utilize distributed training capabilities",
                "Speeds up training and leverages multiple GPUs/TPUs",
                "Configure Trainer for distributed training using distributed or TPU settings",
                node,
                file_path
            )

    def check_mixed_precision_training(self, node: nodes.Module, file_path: str):
        fp16_used = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg == 'fp16' and keyword.value.value for keyword in assign.value.keywords):
                    fp16_used = True
                    break
        if not fp16_used:
            self.add_smell(
                "Mixed precision training not enabled",
                "Use mixed precision training to improve performance",
                "Accelerates training and reduces memory usage",
                "Enable mixed precision training using fp16 parameter in Trainer",
                node,
                file_path
            )

    def check_gradient_accumulation(self, node: nodes.Module, file_path: str):
        gradient_accumulation = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg == 'gradient_accumulation_steps' and keyword.value.value > 1 for keyword in assign.value.keywords):
                    gradient_accumulation = True
                    break
        if not gradient_accumulation:
            self.add_smell(
                "Gradient accumulation not configured",
                "Implement gradient accumulation for large batch sizes",
                "Allows training with larger effective batch sizes and improves convergence",
                "Set gradient_accumulation_steps in Trainer configuration",
                node,
                file_path
            )

    def check_learning_rate_scheduling(self, node: nodes.Module, file_path: str):
        lr_scheduler_used = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg == 'learning_rate_scheduler' for keyword in assign.value.keywords):
                    lr_scheduler_used = True
                    break
        if not lr_scheduler_used:
            self.add_smell(
                "Learning rate scheduler not detected",
                "Use learning rate schedulers to dynamically adjust learning rate",
                "Optimizes training process and enhances model performance",
                "Configure learning_rate_scheduler in Trainer or use transformers built-in schedulers",
                node,
                file_path
            )

    def check_early_stopping(self, node: nodes.Module, file_path: str):
        early_stopping_used = False
        for call in node.nodes_of_class(nodes.Call):
            if 'EarlyStoppingCallback' in call.func.as_string():
                early_stopping_used = True
                break
        if not early_stopping_used:
            self.add_smell(
                "Early stopping not implemented",
                "Implement early stopping to avoid overfitting",
                "Prevents overfitting and reduces unnecessary training time",
                "Use EarlyStoppingCallback in Trainer configuration",
                node,
                file_path
            )

    def generate_report(self) -> str:
        report = "Hugging Face Code Smell Report\n==============================\n\n"
        smell_counts = {}
        for i, smell in enumerate(self.smells, 1):
            if smell['smell'] not in smell_counts:
                smell_counts[smell['smell']] = 0
            smell_counts[smell['smell']] += 1

            report += f"{i}. Smell: {smell['smell']}\n"
            report += f"   File: {smell['file_path']}\n"
            
            # Only show line number if it's not 0
            if smell['line_number'] != 0:
                report += f"   Line: {smell['line_number']}\n"
            
            # Only include code snippet if it's 3 lines or fewer
            code_lines = smell['code_snippet'].strip().split('\n')
            if len(code_lines) <= 3:
                report += f"   Code Snippet:\n{smell['code_snippet']}\n"
            
            report += f"   How to Fix: {smell['how_to_fix']}\n"
            report += f"   Benefits: {smell['benefits']}\n"
            report += f"   Strategies: {smell['strategies']}\n\n"

        report += "Smell Counts:\n"
        for smell, count in smell_counts.items():
            report += f"  {smell}: {count}\n"
        report += f"\nTotal smells detected: {len(self.smells)}"
        return report

