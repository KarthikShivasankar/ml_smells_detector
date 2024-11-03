import astroid
from astroid import nodes
from typing import List, Dict, Any
import sys

class HuggingFaceSmellDetector:
    """A detector class that identifies common code smells in Hugging Face Transformers code.
    
    This detector analyzes Python code that uses the Hugging Face Transformers library and identifies
    potential issues and best practices violations related to model training, data processing,
    and performance optimization.
    """

    def __init__(self):
        self.smells: List[Dict[str, Any]] = []

    def detect_smells(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a Python file for Hugging Face-related code smells.

        Args:
            file_path: Path to the Python file to analyze.

        Returns:
            List of dictionaries containing detected code smells and their details.
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            # Check if 'transformers' is imported
            if self.is_framework_used(module, 'transformers'):
                self.visit_module(module, file_path)
            else:
                print(f"Skipping Hugging Face smell detection for {file_path}: 'transformers' not imported", file=sys.stderr)
        except astroid.exceptions.AstroidSyntaxError as e:
            print(f"Error parsing {file_path}: {str(e)}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {str(e)}", file=sys.stderr)
        return self.smells

    def is_framework_used(self, node: nodes.Module, framework: str) -> bool:
        """Check if a specific framework is imported in the module.

        Args:
            node: AST node representing the module
            framework: Name of the framework to check for

        Returns:
            True if the framework is imported, False otherwise
        """
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                if any(name == framework for name, _ in import_node.names):
                    return True
            elif isinstance(import_node, nodes.ImportFrom):
                if import_node.modname == framework:
                    return True
        return False

    def visit_module(self, node: nodes.Module, file_path: str):
        """Visit a module node and run all smell detection checks.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
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
        """Add a detected code smell to the results.

        Args:
            smell: Description of the code smell
            fix: How to fix the issue
            benefits: Benefits of fixing the issue
            strategies: Specific strategies to implement the fix
            node: AST node where the smell was detected
            file_path: Path to the file containing the smell
        """
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
        """Check if model versions are explicitly specified when loading pre-trained models.
        
        Detects cases where models are loaded without version tags, which could lead to
        reproducibility issues.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in node.nodes_of_class(nodes.Call):
            if ('from_pretrained' in call.func.as_string() and 
                ('AutoModel' in call.func.as_string() or 
                 'PreTrainedModel' in call.func.as_string())):
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
        """Check if tokenizer caching is enabled when loading tokenizers.
        
        Detects cases where tokenizers are loaded without caching configuration, which
        could lead to unnecessary re-downloads and slower loading times.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in node.nodes_of_class(nodes.Call):
            if ('from_pretrained' in call.func.as_string() and 
                ('AutoTokenizer' in call.func.as_string() or 
                 'PreTrainedTokenizer' in call.func.as_string())):
                if not any(keyword.arg in ['cache_dir', 'local_files_only'] 
                          for keyword in call.keywords):
                    self.add_smell(
                        "Tokenizer caching not used",
                        "Cache tokenizers to avoid re-downloading",
                        "Reduces loading time and network dependency",
                        "Use cache_dir parameter when loading tokenizers",
                        call,
                        file_path
                    )

    def check_model_caching(self, node: nodes.Module, file_path: str):
        """Check if model caching is enabled when loading models.
        
        Detects cases where models are loaded without caching configuration, which
        could lead to unnecessary re-downloads and slower loading times.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in node.nodes_of_class(nodes.Call):
            if ('from_pretrained' in call.func.as_string() and 
                ('AutoModel' in call.func.as_string() or 
                 'PreTrainedModel' in call.func.as_string())):
                if not any(keyword.arg in ['cache_dir', 'local_files_only'] 
                          for keyword in call.keywords):
                    self.add_smell(
                        "Model caching not used",
                        "Cache models to avoid re-downloading",
                        "Improves loading efficiency and reduces network dependency",
                        "Use cache_dir parameter when loading models",
                        call,
                        file_path
                    )

    def check_deterministic_tokenization(self, node: nodes.Module, file_path: str):
        """Check if tokenization parameters are explicitly specified.
        
        Detects cases where tokenization settings are not explicitly defined,
        which could lead to inconsistent preprocessing across runs.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in node.nodes_of_class(nodes.Call):
            if ('from_pretrained' in call.func.as_string() and 
                ('AutoTokenizer' in call.func.as_string() or 
                 'PreTrainedTokenizer' in call.func.as_string())):
                deterministic_params = [
                    'do_lower_case', 'strip_accents', 'truncation', 
                    'padding', 'max_length', 'return_tensors'
                ]
                if not any(keyword.arg in deterministic_params for keyword in call.keywords):
                    self.add_smell(
                        "Deterministic tokenization settings not specified",
                        "Use consistent tokenization settings",
                        "Ensures reproducible pre-processing and consistent model inputs",
                        "Set tokenization parameters explicitly when loading tokenizers",
                        call,
                        file_path
                    )

    def check_efficient_data_loading(self, node: nodes.Module, file_path: str):
        """Check if efficient data loading techniques are being used.
        
        Detects cases where standard data loading is used instead of optimized
        methods like datasets library or DataLoader.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        datasets_imported = any('datasets' in import_node.names[0][0] 
                              for import_node in node.nodes_of_class(nodes.ImportFrom))
        
        efficient_patterns = [
            'load_dataset',
            'Dataset.from_',
            'DataLoader',
            'IterableDataset'
        ]
        
        has_efficient_loading = any(
            pattern in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
            for pattern in efficient_patterns
        )
        
        if not (datasets_imported or has_efficient_loading):
            self.add_smell(
                "Efficient data loading not detected",
                "Use efficient data loading techniques",
                "Enhances data processing speed and model training efficiency",
                "Use datasets library for loading and processing data",
                node,
                file_path
            )

    def check_distributed_training(self, node: nodes.Module, file_path: str):
        """Check if distributed training is configured when using training functionality.
        
        Detects cases where training code is present but distributed training
        settings are not configured.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Check if training-related imports exist
        has_training_imports = any(
            'Trainer' in import_node.names[0][0] or 'TrainingArguments' in import_node.names[0][0]
            for import_node in node.nodes_of_class(nodes.ImportFrom)
        )
        
        if not has_training_imports:
            return  # Skip if no training-related imports

        distributed_config = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg in ['local_rank', 'n_gpu', 'distributed_training', 'tpu_num_cores'] 
                      for keyword in assign.value.keywords):
                    distributed_config = True
                    break
        
        # Only report if TrainingArguments is used but without distributed config
        if not distributed_config and self._has_training_arguments(node):
            self.add_smell(
                "Distributed training not configured",
                "Utilize distributed training capabilities",
                "Speeds up training and leverages multiple GPUs/TPUs",
                "Configure Trainer with distributed settings using local_rank, n_gpu, or tpu_num_cores",
                node,
                file_path
            )

    def _has_training_arguments(self, node: nodes.Module) -> bool:
        """Helper method to check if TrainingArguments is actually used in the code"""
        return any(
            isinstance(assign.targets[0], nodes.Name) and 
            assign.targets[0].name == 'TrainingArguments'
            for assign in node.nodes_of_class(nodes.Assign)
        )

    def check_mixed_precision_training(self, node: nodes.Module, file_path: str):
        """Check if mixed precision training is enabled.
        
        Detects cases where training is performed without mixed precision settings,
        which could lead to suboptimal performance and memory usage.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        if not self._has_training_arguments(node):
            return  # Skip if no TrainingArguments used

        fp16_used = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any((keyword.arg == 'fp16' and keyword.value.value) or
                      (keyword.arg == 'bf16' and keyword.value.value) or
                      keyword.arg == 'half_precision_backend'
                      for keyword in assign.value.keywords):
                    fp16_used = True
                    break
        
        if not fp16_used:
            self.add_smell(
                "Mixed precision training not enabled",
                "Use mixed precision training to improve performance",
                "Accelerates training and reduces memory usage",
                "Enable mixed precision training using fp16=True or bf16=True in TrainingArguments",
                node,
                file_path
            )

    def check_gradient_accumulation(self, node: nodes.Module, file_path: str):
        """Check if gradient accumulation is configured for training.
        
        Detects cases where training is performed without gradient accumulation,
        which could be beneficial for handling larger effective batch sizes.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if no TrainingArguments used
        if not self._has_training_arguments(node):
            return

        gradient_accumulation = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg == 'gradient_accumulation_steps' and keyword.value.value > 1 
                      for keyword in assign.value.keywords):
                    gradient_accumulation = True
                    break
        
        # Only report if training configuration is present but gradient accumulation isn't
        if not gradient_accumulation and self._has_training_code(node):
            self.add_smell(
                "Gradient accumulation not configured",
                "Implement gradient accumulation for large batch sizes",
                "Allows training with larger effective batch sizes and improves convergence",
                "Set gradient_accumulation_steps in Trainer configuration",
                node,
                file_path
            )

    def check_learning_rate_scheduling(self, node: nodes.Module, file_path: str):
        """Check if learning rate scheduling is configured.
        
        Detects cases where training is performed without learning rate scheduling,
        which could lead to suboptimal training dynamics.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if no TrainingArguments used
        if not self._has_training_arguments(node):
            return

        lr_scheduler_used = False
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg in ['learning_rate_scheduler', 'lr_scheduler_type'] 
                      for keyword in assign.value.keywords):
                    lr_scheduler_used = True
                    break
        
        # Only report if training configuration is present but lr scheduler isn't
        if not lr_scheduler_used and self._has_training_code(node):
            self.add_smell(
                "Learning rate scheduler not detected",
                "Use learning rate schedulers to dynamically adjust learning rate",
                "Optimizes training process and enhances model performance",
                "Configure lr_scheduler_type in TrainingArguments or use transformers built-in schedulers",
                node,
                file_path
            )

    def check_early_stopping(self, node: nodes.Module, file_path: str):
        """Check if early stopping is implemented in training.
        
        Detects cases where training code is present but early stopping
        mechanisms are not configured, which could lead to overfitting.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if no training-related code is present
        if not self._has_training_code(node):
            return

        early_stopping_used = False
        for call in node.nodes_of_class(nodes.Call):
            if 'EarlyStoppingCallback' in call.func.as_string():
                early_stopping_used = True
                break
            
        # Also check TrainingArguments for early_stopping_* parameters
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name) and assign.targets[0].name == 'TrainingArguments':
                if any(keyword.arg.startswith('early_stopping_') for keyword in assign.value.keywords):
                    early_stopping_used = True
                    break

        if not early_stopping_used:
            self.add_smell(
                "Early stopping not implemented",
                "Implement early stopping to avoid overfitting",
                "Prevents overfitting and reduces unnecessary training time",
                "Use EarlyStoppingCallback or configure early_stopping parameters in TrainingArguments",
                node,
                file_path
            )

    def _has_training_code(self, node: nodes.Module) -> bool:
        """Helper method to check if the code contains training-related elements"""
        training_indicators = [
            'Trainer',
            'TrainingArguments',
            '.train(',
            'optimizer',
            'train_dataset',
            'eval_dataset'
        ]
        
        # Check imports
        has_training_imports = any(
            any(indicator in name for name, _ in import_node.names)
            for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom))
            for indicator in training_indicators
        )
        
        # Check function calls and assignments
        has_training_usage = any(
            any(indicator in node_item.as_string() 
                for indicator in training_indicators)
            for node_item in node.nodes_of_class((nodes.Call, nodes.Assign))
        )
        
        return has_training_imports or has_training_usage

    def generate_report(self) -> str:
        """Generate a formatted report of all detected code smells.

        Returns:
            A string containing the formatted report with all detected smells
            and their counts.
        """
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

    def get_results(self) -> List[Dict[str, str]]:
        """Get the detected smells in a simplified format.

        Returns:
            List of dictionaries containing smell details in a simplified format
            suitable for integration with other tools.
        """
        return [
            {
                'framework': 'Hugging Face',
                'name': smell['smell'],
                'fix': smell['how_to_fix'],
                'benefits': smell['benefits'],
                'location': f"Line {smell['line_number']}" if smell['line_number'] != 0 else ""
            }
            for smell in self.smells
        ]

