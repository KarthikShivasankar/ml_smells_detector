import sys
from typing import Any, Dict, List

import astroid
from astroid import nodes

from ml_code_smell_detector.utils import (
    callee_name,
    get_import_map,
    is_true,
    keyword,
    nodes_of,
    parse_file,
    resolve_call_name,
    run_checks,
    source,
)


class HuggingFaceSmellDetector:
    """A detector class that identifies common code smells in Hugging Face Transformers code.

    This detector analyzes Python code that uses the Hugging Face Transformers library and identifies
    potential issues and best practices violations related to model training, data processing,
    and performance optimization.
    """

    # smell -> (how to fix, benefits, strategies). The smell text is the rule's stable name (see api.list_rules).
    RULES = {
        "Model versioning not specified": (
            "Specify model version when loading pre-trained models",
            "Ensures consistency and reproducibility of results",
            "Pass revision= (a tag or commit hash) to from_pretrained, e.g. from_pretrained(name, revision=\"<commit>\")"),
        "Tokenizer caching not used": (
            "Cache tokenizers to avoid re-downloading",
            "Reduces loading time and network dependency",
            "Rely on the default Hugging Face cache (HF_HOME or cache_dir); do not pass force_download=True"),
        "Model caching not used": (
            "Cache models to avoid re-downloading",
            "Improves loading efficiency and reduces network dependency",
            "Rely on the default Hugging Face cache (HF_HOME or cache_dir); do not pass force_download=True"),
        "Deterministic tokenization settings not specified": (
            "Use consistent tokenization settings",
            "Ensures reproducible pre-processing and consistent model inputs",
            "Pass truncation/padding/max_length explicitly when calling the tokenizer"),
        "Efficient data loading not detected": (
            "Use efficient data loading techniques",
            "Enhances data processing speed and model training efficiency",
            "Use datasets library for loading and processing data"),
        "Distributed training not configured": (
            "Utilize distributed training capabilities",
            "Speeds up training and leverages multiple GPUs/TPUs",
            "Configure TrainingArguments with ddp_backend, fsdp, deepspeed or tpu_num_cores, or launch with torchrun/accelerate"),
        "Mixed precision training not enabled": (
            "Use mixed precision training to improve performance",
            "Accelerates training and reduces memory usage",
            "Enable mixed precision training using fp16=True or bf16=True in TrainingArguments"),
        "Gradient accumulation not configured": (
            "Implement gradient accumulation for large batch sizes",
            "Allows training with larger effective batch sizes and improves convergence",
            "Set gradient_accumulation_steps in Trainer configuration"),
        "Learning rate scheduler not detected": (
            "Use learning rate schedulers to dynamically adjust learning rate",
            "Optimizes training process and enhances model performance",
            "Create one with transformers.get_scheduler (or get_linear_schedule_with_warmup) and call scheduler.step(); "
            "Trainer already applies lr_scheduler_type (default 'linear')"),
        "Early stopping not implemented": (
            "Implement early stopping to avoid overfitting",
            "Prevents overfitting and reduces unnecessary training time",
            "Pass callbacks=[EarlyStoppingCallback(...)] to Trainer with load_best_model_at_end=True and metric_for_best_model"),
    }

    # from_pretrained receivers that are not models (configs, tokenizers, processors)
    NON_MODEL_LOADERS = ('Tokenizer', 'Config', 'Processor', 'FeatureExtractor', 'ImageProcessor')

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
            module = parse_file(file_path)

            # Check if 'transformers' is imported
            if self.is_framework_used(module, 'transformers'):
                self.visit_module(module, file_path)
            else:
                print(
                    f"Skipping Hugging Face smell detection for {file_path}: 'transformers' not imported",
                    file=sys.stderr)
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
        for import_node in nodes_of(node, (nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                if any(name.split('.')[0] == framework for name, _ in import_node.names):
                    return True
            elif isinstance(import_node, nodes.ImportFrom):
                if import_node.modname.split('.')[0] == framework:
                    return True
        return False

    def visit_module(self, node: nodes.Module, file_path: str):
        """Visit a module node and run all smell detection checks.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        run_checks((
            self.check_model_versioning,
            self.check_tokenizer_caching,
            self.check_model_caching,
            self.check_deterministic_tokenization,
            self.check_efficient_data_loading,
            self.check_distributed_training,
            self.check_mixed_precision_training,
            self.check_gradient_accumulation,
            self.check_learning_rate_scheduling,
            self.check_early_stopping,
        ), node, file_path)

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
            "code_snippet": source(node),
            "file_path": file_path
        })

    def _report(self, smell: str, node: nodes.NodeNG, file_path: str):
        self.add_smell(smell, *self.RULES[smell], node, file_path)

    # --- what is being loaded / trained -------------------------------------------------------------

    @staticmethod
    def _loader_class(call: nodes.Call) -> str:
        """'AutoModel' for AutoModel.from_pretrained(...), '' if the call is not a from_pretrained."""
        func = call.func
        if isinstance(func, nodes.Attribute) and func.attrname == 'from_pretrained':
            return source(func.expr).rsplit('.', 1)[-1]
        return ''

    def _is_tokenizer_load(self, call: nodes.Call) -> bool:
        return 'Tokenizer' in self._loader_class(call)

    def _is_model_load(self, call: nodes.Call) -> bool:
        cls = self._loader_class(call)
        return bool(cls) and not any(part in cls for part in self.NON_MODEL_LOADERS)

    @staticmethod
    def _is_local_path(call: nodes.Call) -> bool:
        """from_pretrained('./checkpoint') loads a local directory: there is no Hub revision to pin."""
        arg = call.args[0] if call.args else keyword(call, 'pretrained_model_name_or_path')
        value = arg.value if isinstance(arg, nodes.Const) and isinstance(arg.value, str) else ''
        return value.startswith(('.', '/', '~', '\\')) or value[1:3] in (':\\', ':/')  # last: C:\ drive paths

    @staticmethod
    def _training_args_calls(node: nodes.Module) -> List[nodes.Call]:
        """TrainingArguments(...) / Seq2SeqTrainingArguments(...) calls, however they are bound."""
        return [c for c in nodes_of(node, nodes.Call) if callee_name(c).endswith('TrainingArguments')]

    @staticmethod
    def _passes(call: nodes.Call, names) -> bool:
        """True if any of ``names`` is passed, or **kwargs (which may carry them)."""
        return any(kw.arg in names or kw.arg is None for kw in call.keywords)

    # --- checks -------------------------------------------------------------------------------------

    def check_model_versioning(self, node: nodes.Module, file_path: str):
        """Hub models loaded without revision= (any *Model* class, and pipeline(model=...)). Local paths are skipped.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            # from_pretrained pins a version with revision= (tag, branch or commit); it does not parse 'name@rev'
            if self._is_model_load(call):
                if not self._is_local_path(call) and not self._passes(call, ('revision',)):
                    self._report("Model versioning not specified", call, file_path)
            elif resolve_call_name(call, import_map) == 'transformers.pipeline':
                model = keyword(call, 'model')
                if (isinstance(model, nodes.Const) and isinstance(model.value, str) and
                        not model.value.startswith(('.', '/', '~')) and not self._passes(call, ('revision',))):
                    self._report("Model versioning not specified", call, file_path)

    @staticmethod
    def _bypasses_cache(call: nodes.Call) -> bool:
        """from_pretrained caches by default; only force_download=True actually disables it."""
        return is_true(keyword(call, 'force_download'))

    def check_tokenizer_caching(self, node: nodes.Module, file_path: str):
        """Tokenizers loaded with force_download=True, which bypasses the Hugging Face cache.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in nodes_of(node, nodes.Call):
            if self._is_tokenizer_load(call) and self._bypasses_cache(call):
                self._report("Tokenizer caching not used", call, file_path)

    def check_model_caching(self, node: nodes.Module, file_path: str):
        """Models loaded with force_download=True, which bypasses the Hugging Face cache.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in nodes_of(node, nodes.Call):
            if self._is_model_load(call) and self._bypasses_cache(call):
                self._report("Model caching not used", call, file_path)

    def check_deterministic_tokenization(self, node: nodes.Module, file_path: str):
        """Check if tokenization parameters are explicitly specified.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # truncation/padding/max_length take effect when the tokenizer is *called*, so settings passed to
        # any call count. ponytail: module-level evidence; track the tokenizer variable if this proves too lax.
        call_time_params = {'truncation', 'padding', 'max_length'}
        if any(keyword.arg in call_time_params
               for call in nodes_of(node, nodes.Call) for keyword in call.keywords):
            return
        deterministic_params = ('do_lower_case', 'strip_accents', 'truncation', 'padding', 'max_length', 'return_tensors')
        for call in nodes_of(node, nodes.Call):
            if self._is_tokenizer_load(call) and not self._passes(call, deterministic_params):
                self._report("Deterministic tokenization settings not specified", call, file_path)

    def check_efficient_data_loading(self, node: nodes.Module, file_path: str):
        """Training code that does not use the datasets library or a DataLoader.

        Inference-only scripts are not reported: there is no training data to stream.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        if not self._has_training_code(node):
            return
        datasets_imported = any(
            isinstance(imp, nodes.ImportFrom) and imp.modname.split('.')[0] == 'datasets' or
            isinstance(imp, nodes.Import) and any(name.split('.')[0] == 'datasets' for name, _ in imp.names)
            for imp in nodes_of(node, (nodes.Import, nodes.ImportFrom)))

        efficient_patterns = [
            'load_dataset',
            'Dataset.from_',
            'DataLoader',
            'IterableDataset'
        ]

        has_efficient_loading = any(
            pattern in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for pattern in efficient_patterns
        )

        if not (datasets_imported or has_efficient_loading):
            self._report("Efficient data loading not detected", node, file_path)

    DISTRIBUTED_SETTINGS = ('local_rank', 'ddp_backend', 'ddp_find_unused_parameters', 'fsdp', 'fsdp_config',
                            'deepspeed', 'tpu_num_cores', 'accelerator_config')

    def check_distributed_training(self, node: nodes.Module, file_path: str):
        """TrainingArguments without any distributed setting (DDP, FSDP, DeepSpeed, TPU).

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in self._training_args_calls(node):
            if not self._passes(call, self.DISTRIBUTED_SETTINGS):
                self._report("Distributed training not configured", call, file_path)

    def check_mixed_precision_training(self, node: nodes.Module, file_path: str):
        """TrainingArguments without fp16/bf16 (a non-literal value such as fp16=USE_AMP counts as set).

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in self._training_args_calls(node):
            if not (is_true(keyword(call, 'fp16')) or is_true(keyword(call, 'bf16')) or
                    self._passes(call, ('half_precision_backend', 'fp16_opt_level'))):
                self._report("Mixed precision training not enabled", call, file_path)

    def check_gradient_accumulation(self, node: nodes.Module, file_path: str):
        """TrainingArguments without gradient_accumulation_steps > 1.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for call in self._training_args_calls(node):
            steps = keyword(call, 'gradient_accumulation_steps')
            configured = (steps is not None and not (isinstance(steps, nodes.Const) and steps.value in (0, 1)) or
                          any(kw.arg is None for kw in call.keywords))
            if not configured:
                self._report("Gradient accumulation not configured", call, file_path)

    SCHEDULER_HINTS = ('get_scheduler', 'schedule', 'lr_scheduler', 'LRScheduler', 'OneCycleLR', 'StepLR', 'ReduceLROnPlateau')

    def check_learning_rate_scheduling(self, node: nodes.Module, file_path: str):
        """A custom training loop (loss.backward() + optimizer.step()) without a learning-rate scheduler.

        Trainer is not reported: it always applies TrainingArguments.lr_scheduler_type (default 'linear').

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        calls = list(nodes_of(node, nodes.Call))
        backward = next((c for c in calls if callee_name(c) == 'backward'), None)
        has_step = any(callee_name(c) == 'step' for c in calls)
        if backward is None or not has_step:
            return
        if not any(hint in source(c.func) for c in calls for hint in self.SCHEDULER_HINTS):
            self._report("Learning rate scheduler not detected", backward, file_path)

    def check_early_stopping(self, node: nodes.Module, file_path: str):
        """Trainer-based training without an EarlyStoppingCallback.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        trainers = [c for c in nodes_of(node, nodes.Call) if callee_name(c).endswith('Trainer')]
        if not trainers:
            return
        if not any('EarlyStoppingCallback' in source(call.func) for call in nodes_of(node, nodes.Call)):
            self._report("Early stopping not implemented", trainers[0], file_path)

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
            for import_node in nodes_of(node, (nodes.Import, nodes.ImportFrom))
            for indicator in training_indicators
        )

        # Check function calls and assignments
        has_training_usage = any(
            any(indicator in source(node_item)
                for indicator in training_indicators)
            for node_item in nodes_of(node, (nodes.Call, nodes.Assign))
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
