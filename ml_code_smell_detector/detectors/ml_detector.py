import astroid
from astroid import nodes
from typing import List, Dict, Any
import sys

class ML_SmellDetector:
    """A detector for identifying common code smells in machine learning code.

    This class analyzes Python source code to detect potential issues and anti-patterns
    commonly found in machine learning projects, such as data leakage, improper feature
    scaling, missing cross-validation, and more.
    """

    def __init__(self):
        """Initialize the ML smell detector with empty collections for tracking analysis results."""
        self.smells: List[Dict[str, Any]] = []
        self.imports: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
        self.classes: Dict[str, Any] = {}

    def add_smell(self, smell: str, node: nodes.NodeNG, file_path: str):
        """Add a detected code smell to the collection.

        Args:
            smell: Description of the detected smell
            node: The AST node where the smell was detected
            file_path: Path to the file containing the smell
        """
        self.smells.append({
            "smell": smell,
            "line_number": node.lineno,
            "code_snippet": node.as_string(),
            "file_path": file_path
        })

    def detect_smells(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze a Python file for ML-related code smells.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            List of dictionaries containing detected smell information
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            # Check if any ML-related packages are imported
            ml_packages = ['pandas', 'numpy', 'sklearn', 'tensorflow', 'torch', 'transformers']
            if any(self.is_package_used(module, package) for package in ml_packages):
                self.visit_module(module, file_path)
            else:
                print(f"Skipping ML smell detection for {file_path}: No ML-related packages imported", file=sys.stderr)
        except astroid.exceptions.AstroidSyntaxError as e:
            print(f"Error parsing {file_path}: {str(e)}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {str(e)}", file=sys.stderr)
        return self.smells

    def is_package_used(self, node: nodes.Module, package: str) -> bool:
        """Check if a specific package is imported in the module.

        Args:
            node: AST node representing the module
            package: Name of the package to check for

        Returns:
            True if the package is imported, False otherwise
        """
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                if any(name.split('.')[0] == package for name, _ in import_node.names):
                    return True
            elif isinstance(import_node, nodes.ImportFrom):
                if import_node.modname.split('.')[0] == package:
                    return True
        return False

    def visit_module(self, node: nodes.Module, file_path: str):
        """Run all smell detection checks on a module.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
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
        """Track imports used in the module for later analysis.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                for name, alias in import_node.names:
                    self.imports[alias or name] = name
            elif isinstance(import_node, nodes.ImportFrom):
                for name, alias in import_node.names:
                    self.imports[alias or name] = f"{import_node.modname}.{name}"

    def check_data_leakage(self, node: nodes.Module, file_path: str):
        """Detect potential data leakage from preprocessing before train-test split.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        for func in node.nodes_of_class(nodes.FunctionDef):
            preprocessing_before_split = False
            train_test_split_pos = -1
            last_preprocessing_pos = -1
            
            # Get all Call nodes in order of appearance
            for call in func.nodes_of_class(nodes.Call):
                if call.func.as_string().endswith(('fit', 'fit_transform')):
                    preprocessing_before_split = True
                    last_preprocessing_pos = call.lineno
                elif 'train_test_split' in call.func.as_string():
                    train_test_split_pos = call.lineno
            
            # Only flag if preprocessing occurs before train_test_split in the same function
            if preprocessing_before_split and train_test_split_pos > 0:
                if last_preprocessing_pos < train_test_split_pos:
                    self.add_smell("Potential data leakage: Preprocessing applied before train-test split", func, file_path)

    def check_magic_numbers(self, node: nodes.Module, file_path: str):
        """Detect magic numbers in ML-related code.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Common acceptable values that shouldn't trigger warnings
        acceptable_values = {0, 1, -1, 100, 0.5, 2}  # Common ML-related constants
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.value, nodes.Const) and isinstance(assign.value.value, (int, float)):
                # Skip if it's an acceptable value
                if assign.value.value in acceptable_values:
                    continue
                # Skip if the variable name suggests it's a legitimate constant
                if any(assign.targets[0].as_string().lower().startswith(prefix) for prefix in 
                    ['num_', 'size_', 'batch_', 'epoch', 'learning_rate', 'lr_', 'threshold_']):
                    continue
                self.add_smell(f"Magic number detected: {assign.value.value}", assign, file_path)

    def check_feature_scaling(self, node: nodes.Module, file_path: str):
        """Detect inconsistent feature scaling methods across the codebase.

        Looks for multiple different scaling methods (StandardScaler, MinMaxScaler, etc.)
        being used, which could lead to inconsistent results.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        scaling_methods = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
        scaling_detected = False
        inconsistent_scaling = False
        scalers_used = set()
        
        for call in node.nodes_of_class(nodes.Call):
            if any(method in call.func.as_string() for method in scaling_methods):
                scaling_detected = True
                scalers_used.add(call.func.as_string())
                
        # Only raise warning if multiple different scaling methods are used
        if len(scalers_used) > 1:
            self.add_smell(f"Inconsistent scaling methods detected: {', '.join(scalers_used)}. Consider using the same scaler across the pipeline.", call, file_path)

    def check_cross_validation(self, node: nodes.Module, file_path: str):
        """Check if cross-validation is properly implemented in model training.

        Detects if common cross-validation methods (KFold, cross_val_score, etc.)
        are missing in model training code.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        cv_methods = ['cross_val_score', 'KFold', 'cross_validate', 'GridSearchCV', 'RandomizedSearchCV', 'TimeSeriesSplit']
        cv_detected = False
        is_training_file = False
        
        # Skip if this is likely not a main training file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'preprocessing', 'visualization', 
            'evaluate', 'predict', 'inference', 'deploy'
        ]):
            return

        # Check imports to see if this is likely a training file
        training_imports = {'sklearn.model_selection', 'sklearn.linear_model', 'sklearn.ensemble',
                          'tensorflow', 'torch', 'xgboost', 'lightgbm'}
        has_training_imports = False
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(imp in import_node.modname for imp in training_imports):
                    has_training_imports = True
                    break

        if not has_training_imports:
            return

        for call in node.nodes_of_class(nodes.Call):
            # Check if file contains model training code
            if any(method in call.func.as_string() for method in ['fit', 'train', 'compile']):
                # Skip if it's in a test method
                if any(ancestor.name.startswith('test_') for ancestor in call.node_ancestors() 
                      if isinstance(ancestor, nodes.FunctionDef)):
                    continue
                is_training_file = True
            
            # Check for CV methods
            if any(method in call.func.as_string() for method in cv_methods):
                cv_detected = True
                break
            
            # Also check for custom CV implementations
            if ('split' in call.func.as_string() and 
                any(val in call.as_string() for val in ['fold', 'cv', 'validation'])):
                cv_detected = True
                break
        
        # Only raise warning if it's a substantial training file (has multiple model-related calls)
        model_related_calls = sum(1 for call in node.nodes_of_class(nodes.Call) 
                                if any(term in call.func.as_string() 
                                      for term in ['fit', 'train', 'predict', 'score']))
        
        if (is_training_file and not cv_detected and model_related_calls >= 2 
            and not any(term in file_path.lower() for term in ['quick', 'example', 'demo'])):
            self.add_smell(
                "Cross-validation not detected in model training code. "
                "Consider using cross-validation for more robust evaluation.", 
                node, file_path
            )

    def check_imbalanced_dataset(self, node: nodes.Module, file_path: str):
        """Check if imbalanced dataset handling techniques are used in classification tasks.

        Looks for common techniques like SMOTE, class weights, or stratification
        when dealing with classification problems.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a main training file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'visualization', 'evaluate',
            'predict', 'inference', 'deploy', 'preprocess'
        ]):
            return

        balance_methods = [
            'SMOTE', 'class_weight', 'StratifiedKFold', 'RandomOverSampler', 
            'RandomUnderSampler', 'sample_weight', 'balanced_accuracy',
            'WeightedRandomSampler', 'BalancedBaggingClassifier'
        ]
        imbalance_handling = False
        is_classification = False
        has_data_processing = False
        
        # Check imports first
        classification_imports = {
            'sklearn.linear_model', 'sklearn.ensemble', 'sklearn.svm',
            'sklearn.tree', 'sklearn.naive_bayes', 'xgboost', 'lightgbm'
        }
        has_classification_imports = False
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(imp in import_node.modname for imp in classification_imports):
                    has_classification_imports = True
                    break

        if not has_classification_imports:
            return
        
        # Check if this is a classification task
        for call in node.nodes_of_class(nodes.Call):
            if any(clf in call.func.as_string() for clf in [
                'Classifier', 'LogisticRegression', 'SVC', 'RandomForestClassifier',
                'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier',
                'DecisionTreeClassifier', 'KNeighborsClassifier'
            ]):
                is_classification = True
            
            # Check for data processing/analysis that might indicate class distribution checks
            if any(term in call.func.as_string() for term in [
                'value_counts', 'unique', 'hist', 'countplot', 'distribution',
                'balance_ratio', 'class_distribution'
            ]):
                has_data_processing = True
            
            # Check for imbalance handling methods
            if any(method in call.func.as_string() for method in balance_methods):
                imbalance_handling = True
                break
            
            # Check for custom handling in strings (e.g., parameter names)
            if isinstance(call.func, nodes.Attribute):
                if any(term in str(call.args) + str(call.keywords) for term in [
                    'weight', 'balanced', 'stratif'
                ]):
                    imbalance_handling = True
                    break
        
        # Count model-related calls to ensure it's a substantial training file
        model_related_calls = sum(1 for call in node.nodes_of_class(nodes.Call) 
                                if any(term in call.func.as_string() 
                                      for term in ['fit', 'train', 'predict', 'score']))
        
        # Only raise warning if:
        # 1. It's a classification task
        # 2. No imbalance handling detected
        # 3. Has multiple model-related calls
        # 4. Has data processing (suggesting actual data analysis)
        # 5. Not a quick example/demo
        if (is_classification and not imbalance_handling and 
            model_related_calls >= 2 and has_data_processing and
            not any(term in file_path.lower() for term in ['quick', 'example', 'demo'])):
            
            self.add_smell(
                "No imbalanced dataset handling detected in classification task. "
                "Consider techniques like SMOTE or class weights if dealing with imbalanced data.", 
                node, file_path
            )

    def check_feature_selection(self, node: nodes.Module, file_path: str):
        """Detect feature selection practices and validate their implementation.

        Ensures feature selection is performed with proper validation strategy
        to avoid selection bias.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a feature selection file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'visualization', 
            'predict', 'inference', 'deploy', 'evaluate'
        ]):
            return

        feature_selection_methods = [
            'SelectKBest', 'RFE', 'SelectFromModel', 'PCA', 'VarianceThreshold',
            'mutual_info', 'chi2', 'f_classif', 'SelectPercentile',
            'GenericUnivariateSelect', 'RFECV', 'SequentialFeatureSelector'
        ]
        validation_methods = [
            'cross_val_score', 'train_test_split', 'GridSearchCV',
            'RandomizedSearchCV', 'KFold', 'StratifiedKFold',
            'TimeSeriesSplit', 'cross_validate'
        ]
        feature_selection = False
        validation_detected = False
        has_ml_imports = False
        
        # Check for relevant imports first
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn.feature_selection', 'sklearn.decomposition',
                    'sklearn.model_selection'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        for call in node.nodes_of_class(nodes.Call):
            # Check for feature selection methods
            if any(method in call.func.as_string() for method in feature_selection_methods):
                feature_selection = True
                
            # Check for validation methods
            if any(method in call.func.as_string() for method in validation_methods):
                validation_detected = True
            
            # Check for custom validation in parameter names or strings
            if isinstance(call.func, nodes.Attribute):
                if any(term in str(call.args) + str(call.keywords) for term in 
                    ['valid', 'test', 'split', 'cv', 'fold']):
                    validation_detected = True
        
        # Count substantial ML operations
        ml_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                          if any(term in call.func.as_string() 
                                for term in ['fit', 'transform', 'predict', 'score']))
        
        if (feature_selection and not validation_detected and 
            ml_operations >= 2 and
            not any(term in file_path.lower() for term in ['quick', 'example', 'demo'])):
            self.add_smell(
                "Feature selection detected without clear validation strategy. "
                "Ensure it's applied with proper validation to avoid selection bias.", 
                node, file_path
            )

    def check_metric_selection(self, node: nodes.Module, file_path: str):
        """Validate the choice of evaluation metrics for ML models.

        Ensures appropriate metrics are used for classification and regression tasks,
        warning against using only basic metrics like accuracy.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not an evaluation file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'preprocess', 
            'data', 'feature', 'transform'
        ]):
            return

        metrics = set()
        is_classification = False
        is_regression = False
        has_ml_imports = False
        
        # Check imports first
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn.metrics', 'sklearn.model_selection',
                    'sklearn.linear_model', 'sklearn.ensemble'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        # Determine if it's classification or regression
        for call in node.nodes_of_class(nodes.Call):
            if any(clf in call.func.as_string() for clf in [
                'Classifier', 'LogisticRegression', 'SVC', 'RandomForestClassifier',
                'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier'
            ]):
                is_classification = True
            elif any(reg in call.func.as_string() for reg in [
                'Regressor', 'LinearRegression', 'SVR', 'RandomForestRegressor',
                'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor'
            ]):
                is_regression = True
            
            # Collect metrics
            if any(metric in call.func.as_string() for metric in [
                'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
                'mean_squared_error', 'r2_score', 'mean_absolute_error',
                'roc_auc_score', 'average_precision_score', 'confusion_matrix',
                'classification_report', 'explained_variance_score'
            ]):
                metrics.add(call.func.as_string())
            
            # Check for custom metric implementations
            if isinstance(call.func, nodes.Attribute):
                if any(term in str(call.args) + str(call.keywords) for term in [
                    'metric', 'score', 'evaluation', 'performance'
                ]):
                    metrics.add('custom_metric')

        # Count substantial ML operations
        ml_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                          if any(term in call.func.as_string() 
                                for term in ['fit', 'predict', 'score', 'evaluate']))
        
        # Only raise warnings for specific cases with substantial ML usage
        if ml_operations >= 2 and not any(term in file_path.lower() for term in ['quick', 'example', 'demo']):
            if is_classification and 'accuracy_score' in metrics and len(metrics) == 1:
                self.add_smell(
                    "Only accuracy metric detected for classification. "
                    "Consider adding precision, recall, or F1-score for a more comprehensive evaluation.", 
                    node, file_path
                )
            elif is_regression and 'mean_squared_error' in metrics and len(metrics) == 1:
                self.add_smell(
                    "Only MSE detected for regression. "
                    "Consider adding R2 score or MAE for a more comprehensive evaluation.", 
                    node, file_path
                )

    def check_model_persistence(self, node: nodes.Module, file_path: str):
        """Check model saving practices and associated preprocessing steps.

        Ensures models are saved with their preprocessing steps and proper versioning
        for reproducibility.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a model saving file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'data', 'preprocess',
            'explore', 'analyze', 'visualize'
        ]):
            return

        model_save = False
        preprocessing_save = False
        version_control = False
        has_ml_imports = False
        
        # Check for relevant imports first
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn', 'tensorflow', 'torch', 'joblib', 'pickle'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        # Check for model training/fitting first
        has_model_training = False
        for call in node.nodes_of_class(nodes.Call):
            if any(term in call.func.as_string() for term in ['fit', 'train']):
                has_model_training = True
                break
                
        if not has_model_training:
            return
            
        for call in node.nodes_of_class(nodes.Call):
            # Check for model saving operations
            if any(save in call.func.as_string() for save in [
                'save', 'dump', 'to_pickle', 'save_model', 'save_weights',
                'torch.save', 'joblib.dump'
            ]):
                model_save = True
                
                # Check for preprocessing steps being saved
                if any(prep in call.as_string().lower() for prep in [
                    'scaler', 'encoder', 'preprocessor', 'pipeline',
                    'transform', 'processor', 'tokenizer'
                ]):
                    preprocessing_save = True
                    
                # Check for version control
                if any(ver in call.as_string().lower() for ver in [
                    'version', 'v1', 'v2', 'v3', 'timestamp', 'date',
                    '_v', '.v', 'release'
                ]):
                    version_control = True
                    
                # Check for version control in variable names
                if isinstance(call.func, nodes.Attribute):
                    if any(ver in str(call.args) + str(call.keywords) for ver in [
                        'version', 'timestamp', 'date', 'release'
                    ]):
                        version_control = True
        
        # Only raise warnings if we have substantial model operations
        ml_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                          if any(term in call.func.as_string() 
                                for term in ['fit', 'train', 'predict', 'transform']))
        
        if ml_operations >= 2 and not any(term in file_path.lower() for term in ['quick', 'example', 'demo']):
            if model_save and not preprocessing_save:
                self.add_smell(
                    "Model saving detected without preprocessing steps. "
                    "Remember to save preprocessing steps for proper model deployment.", 
                    node, file_path
                )
            elif model_save and not version_control:
                self.add_smell(
                    "Model saving detected without clear versioning. "
                    "Consider adding version control for model artifacts.", 
                    node, file_path
                )

    def check_reproducibility(self, node: nodes.Module, file_path: str):
        """Check if random seeds are properly set for reproducibility.

        Ensures random seeds are set for all relevant libraries (numpy, random, 
        framework-specific) in ML operations.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a training file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'data', 'preprocess',
            'explore', 'analyze', 'visualize', 'inference'
        ]):
            return

        seed_methods = {
            'random_state', 'seed', 'torch.manual_seed', 'tf.random.set_seed',
            'np.random.seed', 'random.seed', 'cuda.manual_seed',
            'cuda.manual_seed_all', 'tensorflow.random.set_seed'
        }
        seeds_set = set()
        has_ml_operations = False
        has_ml_imports = False
        
        # Check imports first
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn', 'tensorflow', 'torch', 'numpy', 'random'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        for call in node.nodes_of_class(nodes.Call):
            # Check if file contains substantial ML operations
            if any(op in call.func.as_string() for op in [
                'fit', 'train', 'predict', 'transform', 'split',
                'sample', 'shuffle', 'random'
            ]):
                has_ml_operations = True
            
            # Check for seed setting
            for seed_method in seed_methods:
                if seed_method in call.as_string():
                    seeds_set.add(seed_method)
            
            # Check for seed parameters in ML operations
            if isinstance(call.func, nodes.Attribute):
                if any(seed in str(call.args) + str(call.keywords) for seed in [
                    'random_state', 'seed', 'deterministic'
                ]):
                    seeds_set.add('parameter_seed')
        
        # Count substantial ML operations
        ml_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                          if any(term in call.func.as_string() 
                                for term in ['fit', 'train', 'predict', 'transform']))
        
        if (ml_operations >= 2 and has_ml_operations and 
            not any(term in file_path.lower() for term in ['quick', 'example', 'demo'])):
            if not seeds_set:
                self.add_smell(
                    "No random seed setting detected in ML operations. "
                    "Consider setting seeds for reproducibility.", 
                    node, file_path
                )
            elif len(seeds_set) < 2 and any(framework in self.imports for framework in ['tensorflow', 'torch']):
                self.add_smell(
                    "Incomplete seed setting detected. Remember to set seeds for all "
                    "relevant libraries (numpy, random, framework-specific).", 
                    node, file_path
                )

    def check_data_loading(self, node: nodes.Module, file_path: str):
        """Analyze data loading practices for potential issues.

        Checks for proper handling of large datasets, including batch processing
        and file size checks.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a data loading file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'model', 'train',
            'evaluate', 'predict', 'inference'
        ]):
            return

        data_loading_methods = {
            'read_csv', 'load_data', 'read_excel', 'read_parquet',
            'read_json', 'read_sql', 'load_dataset'
        }
        batch_processing_methods = {
            'batch_size', 'generator', 'DataLoader', 'dataset',
            'chunk', 'iterator', 'yield', 'flow_from_directory'
        }
        memory_handling_methods = {
            'dask', 'vaex', 'datatable', 'memory_limit',
            'low_memory', 'nrows', 'usecols'
        }
        
        file_size_check = False
        batch_processing = False
        memory_handling = False
        has_data_imports = False
        
        # Check imports first
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'pandas', 'dask', 'vaex', 'torch.utils.data',
                    'tensorflow.data', 'datasets'
                ]):
                    has_data_imports = True
                    break

        if not has_data_imports:
            return
        
        for call in node.nodes_of_class(nodes.Call):
            if any(method in call.func.as_string() for method in data_loading_methods):
                # Check for file size checks
                if any(check in call.as_string() for check in [
                    'os.path.getsize', 'file_size', 'stat', 'memory_usage'
                ]):
                    file_size_check = True
                    
                # Check for batch processing
                if any(method in call.as_string() for method in batch_processing_methods):
                    batch_processing = True
                    
                # Check for memory handling
                if any(method in call.as_string() for method in memory_handling_methods):
                    memory_handling = True
                    
                # Check for parameters indicating memory consideration
                if isinstance(call.func, nodes.Attribute):
                    if any(param in str(call.args) + str(call.keywords) for param in [
                        'chunksize', 'batch_size', 'iterator', 'memory',
                        'nrows', 'usecols', 'dtype'
                    ]):
                        memory_handling = True
        
        # Only raise warning if we have substantial data loading operations
        data_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                            if any(term in call.func.as_string() 
                                  for term in ['read_', 'load_', 'open']))
        
        if (data_operations >= 2 and 
            not any(term in file_path.lower() for term in ['quick', 'example', 'demo']) and
            not (file_size_check or batch_processing or memory_handling)):
            self.add_smell(
                "Data loading detected without size checks or batch processing. "
                "Consider using generators or batch processing for large datasets.", 
                node, file_path
            )

    def check_unused_features(self, node: nodes.Module, file_path: str):
        """Detect potentially unused features or variables in ML code.

        Identifies variables that are defined but not used, excluding common
        variable names and special prefixes.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a main code file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'conftest', 'setup', '__init__',
            'utils', 'helper', 'config', 'constants'
        ]):
            return

        features = set()
        used_features = set()
        # Expanded list of common variables to ignore
        common_vars = {
            'self', 'i', 'j', 'k', 'x', 'y', 'X', 'y', 'df', 'data',
            'model', 'clf', 'reg', 'pred', 'proba', 'score',
            'train', 'test', 'val', 'valid', 'result', 'output',
            'input', 'params', 'args', 'kwargs', 'config', 'options'
        }
        
        # Skip prefixes for variables that are commonly used in specific ways
        skip_prefixes = {
            '_', 'temp_', 'tmp_', 'test_', 'debug_', 'log_',
            'cache_', 'old_', 'new_', 'raw_', 'processed_'
        }
        
        # Skip suffixes that indicate special usage
        skip_suffixes = {
            '_id', '_idx', '_index', '_key', '_val', '_list',
            '_dict', '_map', '_set', '_array', '_df', '_series'
        }

        has_ml_imports = False
        # Check for ML-related imports
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn', 'tensorflow', 'torch', 'pandas', 'numpy'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name):
                name = assign.targets[0].name
                # Skip if name matches any exclusion criteria
                if (name not in common_vars and 
                    not any(name.startswith(prefix) for prefix in skip_prefixes) and
                    not any(name.endswith(suffix) for suffix in skip_suffixes)):
                    features.add(name)
        
        # Collect used features from various contexts
        for name in node.nodes_of_class(nodes.Name):
            used_features.add(name.name)
            
        # Check for usage in attributes
        for attr in node.nodes_of_class(nodes.Attribute):
            if isinstance(attr.expr, nodes.Name):
                used_features.add(attr.expr.name)
        
        unused = features - used_features - common_vars
        if unused:
            # Additional checks for usage in various contexts
            for node_type in [nodes.Const, nodes.Dict, nodes.List, nodes.Set]:
                for item in node.nodes_of_class(node_type):
                    if isinstance(item.value, str):
                        unused = unused - {feat for feat in unused if feat in item.value}
            
            # Check for usage in f-strings
            for string in node.nodes_of_class(nodes.JoinedStr):
                unused = unused - {feat for feat in unused if feat in string.as_string()}
            
            # Only report if we have substantial ML operations
            ml_operations = sum(1 for call in node.nodes_of_class(nodes.Call) 
                              if any(term in call.func.as_string() 
                                    for term in ['fit', 'predict', 'transform']))
            
            if unused and ml_operations >= 2:
                self.add_smell(
                    f"Potentially unused features detected: {', '.join(unused)}. "
                    "Verify if these are actually needed.", 
                    node, file_path
                )

    def check_overfit_prone_practices(self, node: nodes.Module, file_path: str):
        """Detect practices that might lead to overfitting.

        Checks feature engineering functions for proper train/test separation
        to avoid data leakage.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a feature engineering file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'utils', 'helper', 'config', 'constants',
            'visualization', 'plotting', 'display', 'logging'
        ]):
            return

        has_ml_imports = False
        # Check for ML-related imports
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'sklearn', 'pandas', 'numpy', 'tensorflow', 'torch'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return

        for func in node.nodes_of_class(nodes.FunctionDef):
            # Check if it's a feature engineering function
            if any(term in func.name.lower() for term in [
                'feature', 'transform', 'process', 'engineer', 'prepare'
            ]):
                # Skip if it's clearly safe
                if any(safe_term in func.name.lower() for safe_term in [
                    'train', 'fit', 'training_only', 'train_set',
                    'single', 'one', 'individual', 'row'
                ]):
                    continue
                
                # Check function body for proper data handling
                has_safe_handling = False
                for call in func.nodes_of_class(nodes.Call):
                    if any(term in call.as_string().lower() for term in [
                        'train_test_split', 'validation', 'train_data', 'test_data',
                        'fit_transform', 'transform', 'partial_fit', 'single_transform'
                    ]):
                        has_safe_handling = True
                        break
                
                # Check for parameter names indicating proper handling
                for arg in func.args.args:
                    if any(term in arg.name.lower() for term in [
                        'train', 'test', 'valid', 'single', 'row'
                    ]):
                        has_safe_handling = True
                        break
                
                # Only warn if we have substantial data operations
                data_operations = sum(1 for call in func.nodes_of_class(nodes.Call) 
                                   if any(term in call.func.as_string() 
                                         for term in ['fit', 'transform', 'process']))
                
                if not has_safe_handling and data_operations >= 2:
                    self.add_smell(
                        "Feature engineering function detected without clear train/test separation. "
                        "Ensure it's not applied to the entire dataset to avoid data leakage.", 
                        func, file_path
                    )

    def check_error_handling(self, node: nodes.Module, file_path: str):
        """Check for proper error handling in critical ML operations.

        Ensures try-except blocks or validation checks are present for data
        operations and model-related tasks.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip if this is likely not a main code file
        if any(pattern in file_path.lower() for pattern in [
            'test_', 'conftest', 'setup', '__init__',
            'utils', 'helper', 'config', 'constants',
            'visualization', 'plotting', 'display'
        ]):
            return

        has_data_operations = False
        has_error_handling = False
        critical_operations = [
            'read_csv', 'load_data', 'open', 'fit', 'predict', 
            'transform', 'save', 'dump', 'to_pickle', 'load_model',
            'read_excel', 'read_json', 'read_sql'
        ]
        
        has_ml_imports = False
        # Check for relevant imports
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.ImportFrom):
                if any(pkg in import_node.modname for pkg in [
                    'pandas', 'sklearn', 'tensorflow', 'torch',
                    'pickle', 'joblib'
                ]):
                    has_ml_imports = True
                    break

        if not has_ml_imports:
            return
        
        # Check for critical operations
        critical_op_count = 0
        for call in node.nodes_of_class(nodes.Call):
            if any(op in call.func.as_string() for op in critical_operations):
                has_data_operations = True
                critical_op_count += 1
        
        if has_data_operations:
            # Check for different types of error handling
            for block in node.nodes_of_class((nodes.Try, nodes.ExceptHandler)):
                has_error_handling = True
                break
            
            # Check for validation checks
            for if_block in node.nodes_of_class(nodes.If):
                if any(check in if_block.as_string().lower() for check in [
                    'isinstance', 'isfile', 'exists', 'shape', 'empty', 'null',
                    'none', 'isnull', 'isna', 'type', 'hasattr', 'in',
                    'validate', 'check', 'verify'
                ]):
                    has_error_handling = True
                    break
            
            # Check for assertion statements
            for assert_node in node.nodes_of_class(nodes.Assert):
                has_error_handling = True
                break
            
            if not has_error_handling and critical_op_count >= 2:
                self.add_smell(
                    "No error handling detected in data processing. "
                    "Consider adding try-except blocks or validation checks for robustness.", 
                    node, file_path
                )

    def check_hardcoded_filepaths(self, node: nodes.Module, file_path: str):
        """Detect hardcoded file paths in the code.

        Identifies hardcoded paths that should be moved to configuration files
        or environment variables, excluding common development paths.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Common acceptable patterns
        acceptable_patterns = [
            './test/', './tests/', 
            '../test/', '../tests/',
            'fixtures/', 'data/test/',
            '__pycache__', '.git/',
            'venv/', 'env/'
        ]
        
        config_vars = set()
        # First collect any config/environment variables
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Name):
                if any(config_term in assign.targets[0].name.lower() for config_term in 
                    ['path', 'dir', 'folder', 'file', 'config']):
                    config_vars.add(assign.targets[0].name)
        
        for string in node.nodes_of_class(nodes.Const):
            if isinstance(string.value, str) and ('/' in string.value or '\\' in string.value):
                # Skip if it's a test/common development path
                if any(pattern in string.value for pattern in acceptable_patterns):
                    continue
                    
                # Skip if it's used in a config/path variable assignment
                if any(config_var in string.scope().locals for config_var in config_vars):
                    continue
                    
                self.add_smell(
                    f"Hardcoded file path detected: {string.value}. "
                    "Consider using configuration files or environment variables.", 
                    string, file_path
                )

    def check_documentation(self, node: nodes.Module, file_path: str):
        """Check for proper documentation in ML-related code.

        Ensures functions and classes have appropriate docstrings, especially
        for those with parameters or return values.

        Args:
            node: AST node representing the module
            file_path: Path to the file being analyzed
        """
        # Skip certain types of files/functions
        skip_patterns = [
            'test_', 'fixture', 'conftest',
            'setup', 'init', 'main',
            'helper', 'util'
        ]
        
        for func in node.nodes_of_class(nodes.FunctionDef):
            # Skip if it's a simple function (few lines)
            if len(list(func.get_children())) <= 3:
                continue
                
            # Skip if it's a test or utility function
            if any(pattern in func.name.lower() for pattern in skip_patterns):
                continue
                
            if not isinstance(func.doc_node, nodes.Const):
                # Only warn for functions with parameters or return values
                if func.args.args or 'return' in func.as_string():
                    self.add_smell(
                        f"Missing docstring for function: {func.name}. "
                        "Consider adding documentation for parameters and return values.", 
                        func, file_path
                    )
        
        for cls in node.nodes_of_class(nodes.ClassDef):
            # Skip test classes
            if any(pattern in cls.name.lower() for pattern in skip_patterns):
                continue
                
            if not isinstance(cls.doc_node, nodes.Const):
                # Check if class has public methods
                has_public_methods = any(
                    not method.name.startswith('_') 
                    for method in cls.mymethods()
                )
                if has_public_methods:
                    self.add_smell(
                        f"Missing docstring for class: {cls.name}. "
                        "Consider adding class-level documentation.", 
                        cls, file_path
                    )

    def generate_report(self) -> str:
        """Generate a human-readable report of all detected smells.

        Returns:
            Formatted string containing the analysis report
        """
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

    def get_results(self) -> List[Dict[str, str]]:
        """Get the analysis results in a structured format.

        Returns:
            List of dictionaries containing smell details and locations
        """
        return [
            {
                'framework': 'General ML',
                'name': smell['smell'],
                'fix': "Not specified",
                'benefits': "Not specified",
                'location': f"Line {smell['line_number']}" if smell['line_number'] != 0 else ""
            }
            for smell in self.smells
        ]