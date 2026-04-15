import astroid
from astroid import nodes
from typing import List, Dict, Any

class ML_SmellDetector:
    def __init__(self):
        self.smells: List[Dict[str, Any]] = []
        self.imports: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        self.classes: Dict[str, Any] = {}

    def add_smell(self, smell: str, node: nodes.NodeNG, file_path: str):
        self.smells.append({
            "smell": smell,
            "line_number": node.lineno,
            "code_snippet": node.as_string(),
            "file_path": file_path
        })

    def detect_smells(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r') as file:
            content = file.read()
        module = astroid.parse(content)
        self.visit_module(module, file_path)
        return self.smells

    def visit_module(self, node: nodes.Module, file_path: str):
        self.check_imports(node, file_path)
        self.check_data_leakage(node, file_path)
        self.check_magic_numbers(node, file_path)
        self.check_feature_scaling(node, file_path)
        self.check_cross_validation(node, file_path)
        self.check_imbalanced_dataset(node, file_path)
        self.check_feature_selection(node, file_path)
        self.check_metric_selection(node, file_path)
        self.check_model_persistence(node, file_path)
        self.check_reproducibility(node, file_path)
        self.check_data_loading(node, file_path)
        self.check_unused_features(node, file_path)
        self.check_overfit_prone_practices(node, file_path)
        self.check_error_handling(node, file_path)
        self.check_hardcoded_filepaths(node, file_path)
        self.check_documentation(node, file_path)
        

    def check_imports(self, node: nodes.Module, file_path: str):
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                for name, alias in import_node.names:
                    self.imports[alias or name] = name
            elif isinstance(import_node, nodes.ImportFrom):
                for name, alias in import_node.names:
                    self.imports[alias or name] = f"{import_node.modname}.{name}"

    def check_data_leakage(self, node: nodes.Module, file_path: str):
        preprocessing_before_split = False
        train_test_split = False
        
        for func in node.nodes_of_class(nodes.FunctionDef):
            if any(call.func.as_string().endswith(('fit', 'fit_transform')) for call in func.nodes_of_class(nodes.Call)):
                preprocessing_before_split = True
            if 'train_test_split' in func.as_string():
                train_test_split = True
                
        if preprocessing_before_split and not train_test_split:
            self.add_smell("Potential data leakage: Preprocessing applied before train-test split", func, file_path)

    def check_magic_numbers(self, node: nodes.Module, file_path: str):
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.value, nodes.Const) and isinstance(assign.value.value, (int, float)):
                self.add_smell(f"Magic number detected: {assign.value.value}", assign, file_path)

    def check_feature_scaling(self, node: nodes.Module, file_path: str):
        scaling_methods = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
        for call in node.nodes_of_class(nodes.Call):
            if any(method in call.func.as_string() for method in scaling_methods):
                # This is a simplistic check. In a real scenario, you'd want to ensure consistent use across features.
                self.add_smell("Feature scaling detected. Ensure consistent application across features and datasets.", call, file_path)

    def check_cross_validation(self, node: nodes.Module, file_path: str):
        cv_detected = False
        for call in node.nodes_of_class(nodes.Call):
            if 'cross_val_score' in call.func.as_string() or 'KFold' in call.func.as_string():
                cv_detected = True
                break
        if not cv_detected:
            self.add_smell("Cross-validation not detected. Consider using cross-validation for more robust evaluation.", call, file_path)

    def check_imbalanced_dataset(self, node: nodes.Module, file_path: str):
        imbalance_handling = False
        for call in node.nodes_of_class(nodes.Call):
            if any(method in call.func.as_string() for method in ['SMOTE', 'class_weight', 'StratifiedKFold']):
                imbalance_handling = True
                break
        if not imbalance_handling:
            self.add_smell("No imbalanced dataset handling detected. Consider techniques like SMOTE or class weights if dealing with imbalanced data.", call, file_path)

    def check_feature_selection(self, node: nodes.Module, file_path: str):
        feature_selection = False
        for call in node.nodes_of_class(nodes.Call):
            if any(method in call.func.as_string() for method in ['SelectKBest', 'RFE', 'SelectFromModel']):
                feature_selection = True
                break
        if feature_selection:
            self.add_smell("Feature selection detected. Ensure it's applied with proper validation.", call, file_path)

    def check_metric_selection(self, node: nodes.Module, file_path: str):
        metrics = set()
        for call in node.nodes_of_class(nodes.Call):
            if 'metric' in call.func.as_string().lower():
                metrics.add(call.func.as_string())
        if 'accuracy_score' in metrics and len(metrics) == 1:
            self.add_smell("Only accuracy metric detected. Consider using additional metrics for a more comprehensive evaluation.", call, file_path)

    def check_model_persistence(self, node: nodes.Module, file_path: str):
        model_save = False
        for call in node.nodes_of_class(nodes.Call):
            if 'save' in call.func.as_string() or 'dump' in call.func.as_string():
                model_save = True
                break
        if model_save:
            self.add_smell("Model saving detected. Ensure proper versioning and saving of preprocessing steps.", call, file_path)

    def check_reproducibility(self, node: nodes.Module, file_path: str):
        seed_set = False
        for call in node.nodes_of_class(nodes.Call):
            if 'random_state' in call.as_string() or 'seed' in call.as_string():
                seed_set = True
                break
        if not seed_set:
            self.add_smell("No random seed setting detected. Consider setting seeds for reproducibility.", call, file_path)

    def check_data_loading(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'read_csv' in call.func.as_string() or 'load_data' in call.func.as_string():
                self.add_smell("Data loading detected. For large datasets, consider using generators or batch processing.", call, file_path)

    def check_unused_features(self, node: nodes.Module, file_path: str):
        # This is a simplistic check. A more comprehensive check would involve tracking feature usage across the entire pipeline.
        features = set()
        used_features = set()
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name):
                features.add(assign.targets[0].name)
        for name in node.nodes_of_class(nodes.Name):
            used_features.add(name.name)
        unused = features - used_features
        if unused:
            self.add_smell(f"Potentially unused features detected: {', '.join(unused)}", assign, file_path)

    def check_overfit_prone_practices(self, node: nodes.Module, file_path: str):
        for func in node.nodes_of_class(nodes.FunctionDef):
            if 'feature_engineering' in func.name.lower() and 'train' not in func.name.lower():
                self.add_smell("Feature engineering function detected. Ensure it's not applied to the entire dataset to avoid data leakage.", func, file_path)

    def check_error_handling(self, node: nodes.Module, file_path: str):
        try_except = False
        for block in node.nodes_of_class(nodes.Try):
            try_except = True
            break
        if not try_except:
            self.add_smell("No error handling detected in data processing. Consider adding try-except blocks for robustness.", node, file_path)
            
    def check_hardcoded_filepaths(self, node: nodes.Module, file_path: str):
        for string in node.nodes_of_class(nodes.Const):
            if isinstance(string.value, str) and ('/' in string.value or '\\' in string.value):
                self.add_smell(f"Hardcoded file path detected: {string.value}", string, file_path)

    def check_documentation(self, node: nodes.Module, file_path: str):
        for func in node.nodes_of_class(nodes.FunctionDef):
            if not isinstance(func.doc_node, nodes.Const):
                self.add_smell(f"Missing docstring for function: {func.name}", func, file_path)
        for cls in node.nodes_of_class(nodes.ClassDef):
            if not isinstance(cls.doc_node, nodes.Const):
                self.add_smell(f"Missing docstring for class: {cls.name}", cls, file_path)

    def generate_report(self) -> str:
        report = "General ML Code Smell Report\n============================\n\n"
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
            report += "\n"

        report += "Smell Counts:\n"
        for smell, count in smell_counts.items():
            report += f"  {smell}: {count}\n"
        report += f"\nTotal smells detected: {len(self.smells)}"
        return report

