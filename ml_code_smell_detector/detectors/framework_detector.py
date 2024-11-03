import astroid
from astroid import nodes
from typing import List, Dict, Any
import sys

class FrameworkSpecificSmellDetector:
    """A detector for identifying code smells specific to ML frameworks like Pandas, NumPy, Scikit-learn, PyTorch and TensorFlow.
    
    This class analyzes Python code to detect common anti-patterns and suboptimal implementations
    when using popular machine learning frameworks. It provides suggestions for improvements
    and best practices.
    """

    def __init__(self):
        self.smells: List[Dict[str, Any]] = []
        self.framework_smells = self.get_smells()

    def detect_smells(self, file_path: str) -> List[Dict[str, str]]:
        """Analyze a Python file for framework-specific code smells.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            List of dictionaries containing detected code smells with details like:
            - framework: The ML framework the smell relates to
            - name: Name of the code smell
            - how_to_fix: Instructions for fixing the issue
            - benefits: Benefits of fixing the issue
            - line_number: Line where the smell was detected
            - code_snippet: The problematic code
            - file_path: Path to the file containing the smell
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            module = astroid.parse(content, module_name=file_path)
            
            if not module:
                print(f"Error: Could not parse module for {file_path}", file=sys.stderr)
                return self.smells

            frameworks_used = self.get_frameworks_used(module)
            if frameworks_used is None:
                print(f"Error: Could not determine frameworks for {file_path}", file=sys.stderr)
                return self.smells
            
            if frameworks_used:
                self.visit_module(module, file_path, frameworks_used)
            else:
                print(f"Skipping framework-specific smell detection for {file_path}: No relevant frameworks imported", file=sys.stderr)
        except astroid.exceptions.AstroidSyntaxError as e:
            print(f"Error parsing {file_path}: {str(e)}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error while processing {file_path}: {str(e)}", file=sys.stderr)
        return self.smells

    def get_frameworks_used(self, node: nodes.Module) -> List[str]:
        """Detect which ML frameworks are imported in the code.

        Args:
            node: AST node representing the Python module

        Returns:
            List of framework names found in imports (e.g. ['Pandas', 'NumPy'])
        """
        if not node:
            return []
        
        frameworks = []
        framework_imports = {
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'sklearn': 'ScikitLearn',
            'tensorflow': 'TensorFlow',
            'torch': 'PyTorch'
        }
        
        try:
            for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
                if isinstance(import_node, nodes.Import):
                    for name, _ in import_node.names:
                        if name in framework_imports:
                            frameworks.append(framework_imports[name])
                elif isinstance(import_node, nodes.ImportFrom):
                    if import_node.modname in framework_imports:
                        frameworks.append(framework_imports[import_node.modname])
        except Exception as e:
            print(f"Error in get_frameworks_used: {str(e)}", file=sys.stderr)
            return []
        
        return frameworks

    def visit_module(self, node: nodes.Module, file_path: str, frameworks_used: List[str]):
        """Visit a Python module to detect framework-specific smells.

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed  
            frameworks_used: List of frameworks detected in the code
        """
        if not node or not frameworks_used:
            return
        
        try:
            self.check_imports(node, file_path)
            for framework in frameworks_used:
                if framework:
                    detector_method = getattr(self, f"detect_{framework.lower()}_smells", None)
                    if detector_method:
                        detector_method(node, file_path)
        except Exception as e:
            print(f"Error in visit_module for {file_path}: {str(e)}", file=sys.stderr)

    def add_smell(self, framework: str, smell_name: str, node: nodes.NodeNG, file_path: str):
        """Add a detected code smell to the results.

        Args:
            framework: Name of the framework (e.g. 'Pandas', 'NumPy')
            smell_name: Name of the detected smell
            node: AST node where the smell was found
            file_path: Path to the file containing the smell
        """
        smell = next((s for s in self.framework_smells[framework] if s['name'] == smell_name), None)
        if smell:
            self.smells.append({
                "framework": framework,
                "name": smell['name'],
                "how_to_fix": smell['how_to_fix'],
                "benefits": smell['benefits'],
                "strategies": smell['strategies'],
                "line_number": node.lineno,
                "code_snippet": node.as_string(),
                "file_path": file_path
            })

    # Pandas Detection Methods
    def detect_pandas_smells(self, node: nodes.Module, file_path: str):
        """Detect Pandas-specific code smells like:
        - Unnecessary iteration instead of vectorization
        - Chain indexing
        - Missing merge parameters
        - Inplace operations
        - Inefficient values usage
        - Missing dtype specifications
        - Suboptimal column selection
        - DataFrame modifications in loops

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed
        """
        self.detect_iterrows_usage(node, file_path)
        self.detect_chain_indexing(node, file_path)
        self.detect_merge_parameters(node, file_path)
        self.detect_inplace_operations(node, file_path)
        self.detect_values_usage(node, file_path)
        self.detect_dtype_specification(node, file_path)
        self.detect_column_selection(node, file_path)
        self.detect_dataframe_modification(node, file_path)

    def detect_iterrows_usage(self, node: nodes.Module, file_path: str):
        # Only flag iterrows if better alternatives could be used
        vectorizable_operations = ['sum', 'mean', 'max', 'min', 'apply', 'map']
        for call in node.nodes_of_class(nodes.Call):
            if 'iterrows' in call.func.as_string():
                # Check if the iterrows is used within a loop
                parent = call.parent
                while parent and not isinstance(parent, nodes.For):
                    parent = parent.parent
                if parent and any(op in parent.as_string() for op in vectorizable_operations):
                    self.add_smell('Pandas', 'Unnecessary Iteration', call, file_path)

    def detect_chain_indexing(self, node: nodes.Module, file_path: str):
        for subscript in node.nodes_of_class(nodes.Subscript):
            if isinstance(subscript.value, nodes.Subscript):
                # Only flag if it's not within a with pd.option_context block
                parent = subscript.parent
                while parent and not isinstance(parent, nodes.With):
                    parent = parent.parent
                if not parent or 'option_context' not in parent.as_string():
                    # Check if it's an assignment (more dangerous) vs just access
                    if isinstance(subscript.parent, nodes.Assign):
                        self.add_smell('Pandas', 'Chain Indexing', subscript, file_path)

    def detect_merge_parameters(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if ('merge' in call.func.as_string() and 
                isinstance(call.func, nodes.Attribute) and 
                # Check if it's actually a pandas merge
                ('pd' in call.func.expr.as_string() or 'pandas' in call.func.expr.as_string()) and 
                not any(kw.arg in ['how', 'on', 'validate'] for kw in call.keywords)):
                self.add_smell('Pandas', 'Merge Parameter Checker', call, file_path)

    def detect_inplace_operations(self, node: nodes.Module, file_path: str):
        inplace_operations = ['sort_values', 'fillna', 'drop', 'replace', 'rename']
        for call in node.nodes_of_class(nodes.Call):
            if ('inplace=True' in call.as_string() and 
                any(op in call.func.as_string() for op in inplace_operations)):
                # Check if the result is used later
                parent = call.parent
                while parent and not isinstance(parent, nodes.Module):
                    if isinstance(parent, nodes.Assign):
                        break
                    parent = parent.parent
                if not isinstance(parent, nodes.Assign):
                    self.add_smell('Pandas', 'InPlace Checker', call, file_path)

    def detect_values_usage(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if '.values' in call.as_string():
                # Check if it's used in a context where to_numpy() would be better
                numpy_contexts = ['np.', 'array', 'asarray', 'reshape', 'transpose']
                if any(ctx in call.as_string() for ctx in numpy_contexts):
                    self.add_smell('Pandas', 'DataFrame Conversion Checker', call, file_path)

    def detect_dtype_specification(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if ('read_csv' in call.func.as_string() and 
                # Check if it's actually pandas read_csv
                ('pd' in call.func.as_string() or 'pandas' in call.func.as_string()) and
                not any(kw.arg in ['dtype', 'parse_dates'] for kw in call.keywords)):
                # Check if the DataFrame is used in operations sensitive to dtypes
                parent = call.parent
                dtype_sensitive_ops = ['groupby', 'merge', 'join', 'sort_values', 'arithmetic_operations']
                while parent and not isinstance(parent, nodes.Module):
                    if any(op in parent.as_string() for op in dtype_sensitive_ops):
                        self.add_smell('Pandas', 'Datatype Checker', call, file_path)
                        break
                    parent = parent.parent

    def detect_column_selection(self, node: nodes.Module, file_path: str):
        # Only check if there are actual DataFrame operations
        df_operations = False
        column_selections = False
        
        for subscript in node.nodes_of_class(nodes.Subscript):
            if isinstance(subscript.value, nodes.Name):
                # Look for DataFrame operations
                if any(op in node.as_string() for op in ['DataFrame', 'pd.', 'pandas']):
                    df_operations = True
                    if '[[' in subscript.as_string():
                        column_selections = True
                        break
        
        if df_operations and not column_selections:
            self.add_smell('Pandas', 'Column Selection Checker', node, file_path)

    def detect_dataframe_modification(self, node: nodes.Module, file_path: str):
        for assign in node.nodes_of_class(nodes.Assign):
            if (isinstance(assign.targets[0], nodes.Subscript) and 
                isinstance(assign.targets[0].value, nodes.Name)):
                # Check if it's within a loop and modifying a DataFrame
                parent = assign.parent
                while parent and not isinstance(parent, nodes.For):
                    parent = parent.parent
                
                # Check if it's actually a DataFrame modification
                df_indicators = ['DataFrame', 'pd.', 'pandas']
                if (parent and 
                    any(indicator in assign.as_string() for indicator in df_indicators) and
                    not any(safe_op in assign.as_string() for safe_op in ['loc', 'iloc', 'at', 'iat'])):
                    self.add_smell('Pandas', 'DataFrame Iteration Modification', assign, file_path)

    # NumPy Detection Methods
    def detect_numpy_smells(self, node: nodes.Module, file_path: str):
        """Detect NumPy-specific code smells like:
        - NaN equality comparisons
        - Missing random seeds
        - Inefficient array creation
        - Non-vectorized operations
        - Dtype inconsistencies
        - Broadcasting issues
        - Copy/view confusion
        - Missing axis specifications

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed
        """
        self.detect_nan_equality(node, file_path)
        self.detect_random_seed(node, file_path)
        self.detect_array_creation(node, file_path)
        self.detect_inefficient_operations(node, file_path)
        self.detect_dtype_consistency(node, file_path)
        self.detect_broadcasting_issues(node, file_path)
        self.detect_copy_view_issues(node, file_path)
        self.detect_axis_specification(node, file_path)

    def detect_nan_equality(self, node: nodes.Module, file_path: str):
        for compare in node.nodes_of_class(nodes.Compare):
            if any('np.nan' in getattr(op[0], 'as_string', lambda: '')() for op in compare.ops):
                self.add_smell('NumPy', 'NaN Equality Checker', compare, file_path)

    def detect_random_seed(self, node: nodes.Module, file_path: str):
        # List of numpy random operations to check for
        random_operations = [
            'np.random.rand',
            'np.random.randn',
            'np.random.randint',
            'np.random.choice',
            'np.random.shuffle',
            'np.random.permutation',
            'np.random.normal',
            'np.random.uniform'
        ]
        
        # Find all random operation calls
        random_op_calls = [
            call for call in node.nodes_of_class(nodes.Call)
            if any(op in getattr(call.func, 'as_string', lambda: '')() for op in random_operations)
        ]
        
        # Find all random seed calls
        random_seed_calls = [
            call for call in node.nodes_of_class(nodes.Call)
            if 'np.random.seed' in getattr(call.func, 'as_string', lambda: '')()
        ]
        
        # Only trigger smell if random operations are used without seed
        if random_op_calls and not random_seed_calls:
            self.add_smell('NumPy', 'Randomness Control Checker', node, file_path)

    def detect_array_creation(self, node: nodes.Module, file_path: str):
        """Detect inefficient array creation patterns"""
        for call in node.nodes_of_class(nodes.Call):
            # Check for list to array conversion without dtype
            if ('np.array' in call.func.as_string() and 
                not any(kw.arg == 'dtype' for kw in call.keywords)):
                self.add_smell('NumPy', 'Array Creation Efficiency', call, file_path)
            
            # Check for zeros/ones/empty without dtype
            if any(func in call.func.as_string() for func in ['np.zeros', 'np.ones', 'np.empty']) and \
               not any(kw.arg == 'dtype' for kw in call.keywords):
                self.add_smell('NumPy', 'Array Creation Efficiency', call, file_path)

    def detect_inefficient_operations(self, node: nodes.Module, file_path: str):
        """Detect inefficient numerical operations"""
        for loop in node.nodes_of_class(nodes.For):
            # Check for element-wise operations in loops
            if any(op in loop.as_string() for op in ['np.sum', 'np.mean', 'np.max', 'np.min']):
                self.add_smell('NumPy', 'Vectorization Opportunity', loop, file_path)
        
        # Check for inefficient concatenation
        for call in node.nodes_of_class(nodes.Call):
            if 'np.concatenate' in call.func.as_string():
                parent = call.parent
                while parent and not isinstance(parent, nodes.For):
                    parent = parent.parent
                if parent:  # If concatenate is inside a loop
                    self.add_smell('NumPy', 'Inefficient Concatenation', call, file_path)

    def detect_dtype_consistency(self, node: nodes.Module, file_path: str):
        """Detect potential dtype inconsistency issues"""
        for binop in node.nodes_of_class(nodes.BinOp):
            # Check for mixed integer and float operations
            if (isinstance(binop.left, nodes.Call) and isinstance(binop.right, nodes.Call) and
                'np.' in binop.left.func.as_string() and 'np.' in binop.right.func.as_string()):
                if ('int' in binop.left.func.as_string() and 'float' in binop.right.func.as_string()) or \
                   ('float' in binop.left.func.as_string() and 'int' in binop.right.func.as_string()):
                    self.add_smell('NumPy', 'Dtype Consistency', binop, file_path)

    def detect_broadcasting_issues(self, node: nodes.Module, file_path: str):
        """Detect potential broadcasting issues"""
        for binop in node.nodes_of_class(nodes.BinOp):
            if isinstance(binop.left, nodes.Call) and isinstance(binop.right, nodes.Call):
                # Check for operations between arrays that might have broadcasting issues
                if ('reshape' in binop.left.as_string() or 'reshape' in binop.right.as_string() or
                    'transpose' in binop.left.as_string() or 'transpose' in binop.right.as_string()):
                    self.add_smell('NumPy', 'Broadcasting Risk', binop, file_path)

    def detect_copy_view_issues(self, node: nodes.Module, file_path: str):
        """Detect potential copy/view confusion"""
        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.value, nodes.Subscript):
                # Check for array slicing without explicit copy
                if not any(method in assign.parent.as_string() for method in ['.copy()', 'np.copy']):
                    self.add_smell('NumPy', 'Copy-View Confusion', assign, file_path)

    def detect_axis_specification(self, node: nodes.Module, file_path: str):
        """Detect missing axis specifications in array operations"""
        axis_operations = ['sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'any', 'all']
        for call in node.nodes_of_class(nodes.Call):
            if (any(op in call.func.as_string() for op in axis_operations) and
                'np.' in call.func.as_string() and
                not any(kw.arg == 'axis' for kw in call.keywords) and
                len(call.args) < 2):  # axis can also be specified as positional argument
                self.add_smell('NumPy', 'Missing Axis Specification', call, file_path)

    # Scikit-learn Detection Methods
    def detect_sklearn_smells(self, node: nodes.Module, file_path: str):
        """Detect Scikit-learn specific code smells like:
        - Missing data scaling
        - Not using pipelines
        - Missing cross validation
        - Missing random state
        - Missing verbose mode
        - Using only threshold-dependent metrics
        - Missing unit tests
        - Data leakage risks
        - Missing exception handling

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed
        """
        self.detect_scaling_usage(node, file_path)
        self.detect_pipeline_usage(node, file_path)
        self.detect_cross_validation(node, file_path)
        self.detect_random_state(node, file_path)
        self.detect_verbose_mode(node, file_path)
        self.detect_threshold_metrics(node, file_path)
        self.detect_unit_tests(node, file_path)
        self.detect_data_leakage(node, file_path)
        self.detect_exception_handling(node, file_path)

    def detect_scaling_usage(self, node: nodes.Module, file_path: str):
        scaling_sensitive_estimators = [
            'SVM', 'SVR', 'PCA', 'KMeans', 'NeuralNetwork', 'LogisticRegression'
        ]
        scaling_methods = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
        
        # Check if scaling-sensitive estimators are used
        has_sensitive_estimator = any(
            estimator in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for estimator in scaling_sensitive_estimators
        )
        
        # Check if any scaling is applied
        has_scaling = any(
            method in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for method in scaling_methods
        )
        
        # Only report if using sensitive estimators without scaling
        if has_sensitive_estimator and not has_scaling:
            self.add_smell('ScikitLearn', 'Scaler Missing Checker', node, file_path)

    def detect_pipeline_usage(self, node: nodes.Module, file_path: str):
        # Check if there are multiple preprocessing or model fitting steps
        preprocessing_steps = [
            'StandardScaler', 'MinMaxScaler', 'PCA', 'SelectKBest', 
            'PolynomialFeatures', 'OneHotEncoder', 'LabelEncoder'
        ]
        model_steps = [
            'fit', 'predict', 'transform', 'fit_transform'
        ]
        
        has_preprocessing = any(
            step in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for step in preprocessing_steps
        )
        has_model_steps = any(
            step in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for step in model_steps
        )
        
        # Only suggest Pipeline if multiple steps are present
        if has_preprocessing and has_model_steps and not any(
            'Pipeline' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        ):
            self.add_smell('ScikitLearn', 'Pipeline Checker', node, file_path)

    def detect_cross_validation(self, node: nodes.Module, file_path: str):
        # Check if model training is performed
        training_indicators = ['fit', 'train']
        has_training = any(
            indicator in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for indicator in training_indicators
        )
        
        # Check for any cross-validation technique
        cv_methods = [
            'cross_val_score', 'KFold', 'StratifiedKFold', 
            'cross_validate', 'GridSearchCV', 'RandomizedSearchCV'
        ]
        has_cv = any(
            method in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for method in cv_methods
        )
        
        # Only suggest cross-validation for model training scenarios
        if has_training and not has_cv:
            self.add_smell('ScikitLearn', 'Cross Validation Checker', node, file_path)

    def detect_random_state(self, node: nodes.Module, file_path: str):
        # List of methods that accept random_state
        random_state_methods = [
            'train_test_split', 'KFold', 'RandomForest', 'KMeans',
            'PCA', 'shuffle', 'random_state'
        ]
        
        # Check if any random-dependent operations are used
        random_dependent_calls = [
            call for call in node.nodes_of_class(nodes.Call)
            if any(method in call.func.as_string() for method in random_state_methods)
        ]
        
        # Check if random_state is set for these calls
        for call in random_dependent_calls:
            if not any(
                kw.arg == 'random_state' for kw in call.keywords
            ):
                self.add_smell('ScikitLearn', 'Randomness Control Checker', call, file_path)

    def detect_verbose_mode(self, node: nodes.Module, file_path: str):
        # Only check for verbose in time-consuming operations
        time_consuming_ops = [
            'GridSearchCV', 'RandomizedSearchCV', 'fit', 
            'RandomForest', 'GradientBoosting'
        ]
        
        for call in node.nodes_of_class(nodes.Call):
            if (any(op in call.func.as_string() for op in time_consuming_ops) and
                not any(kw.arg == 'verbose' for kw in call.keywords)):
                self.add_smell('ScikitLearn', 'Verbose Mode Checker', call, file_path)

    def detect_threshold_metrics(self, node: nodes.Module, file_path: str):
        # Only check for classification-related tasks
        classification_indicators = [
            'classifier', 'predict_proba', 'accuracy_score', 
            'precision_score', 'recall_score', 'f1_score',
            'classification_report'
        ]
        
        # Check if it's a classification task
        is_classification = any(
            indicator in node.as_string() 
            for indicator in classification_indicators
        )
        
        # Check for threshold-independent metrics
        threshold_independent_metrics = [
            'roc_auc_score', 'average_precision_score', 
            'precision_recall_curve', 'roc_curve'
        ]
        has_threshold_metrics = any(
            metric in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for metric in threshold_independent_metrics
        )
        
        # Only suggest threshold-independent metrics for classification tasks
        if is_classification and not has_threshold_metrics:
            self.add_smell('ScikitLearn', 'Dependent Threshold Checker', node, file_path)

    def detect_unit_tests(self, node: nodes.Module, file_path: str):
        # Check if this is a source file (not already a test file)
        is_test_file = ('test' in file_path.lower() or 
                       node.as_string().lower().startswith('test'))
        
        # Check for model training or evaluation code
        ml_operations = [
            'fit', 'predict', 'transform', 'score', 
            'cross_val_score', 'GridSearchCV'
        ]
        has_ml_operations = any(
            op in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for op in ml_operations
        )
        
        # Check for testing frameworks
        test_frameworks = [
            'unittest', 'pytest', 'nose', 
            'TestCase', '@test', '@pytest'
        ]
        has_tests = any(
            framework in node.as_string() 
            for framework in test_frameworks
        )
        
        # Only suggest adding tests for non-test ML files without tests
        if not is_test_file and has_ml_operations and not has_tests:
            self.add_smell('ScikitLearn', 'Unit Testing Checker', node, file_path)

    def detect_data_leakage(self, node: nodes.Module, file_path: str):
        # Check for data preprocessing operations
        preprocessing_ops = [
            'fit_transform', 'transform', 'StandardScaler', 
            'MinMaxScaler', 'PCA', 'feature_selection'
        ]
        has_preprocessing = any(
            op in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for op in preprocessing_ops
        )
        
        # Check for model training
        has_model_training = any(
            'fit' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Check for proper train-test splitting
        splitting_methods = [
            'train_test_split', 'KFold', 'StratifiedKFold',
            'GroupKFold', 'TimeSeriesSplit'
        ]
        has_proper_split = any(
            method in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for method in splitting_methods
        )
        
        # Only report if preprocessing and training without proper splitting
        if (has_preprocessing and has_model_training and not has_proper_split):
            self.add_smell('ScikitLearn', 'Data Leakage Checker', node, file_path)

    def detect_exception_handling(self, node: nodes.Module, file_path: str):
        # Check for risky operations that should have exception handling
        risky_operations = [
            'fit', 'predict', 'transform', 'inverse_transform',
            'cross_val_score', 'GridSearchCV', 'load_', 'dump_',
            'pickle', 'joblib'
        ]
        
        # Find all risky operation calls
        risky_calls = [
            call for call in node.nodes_of_class(nodes.Call)
            if any(op in call.func.as_string() for op in risky_operations)
        ]
        
        # Check if these calls are within try-except blocks
        for call in risky_calls:
            parent = call.parent
            within_try_block = False
            
            while parent and not isinstance(parent, nodes.Module):
                if isinstance(parent, nodes.Try):
                    within_try_block = True
                    break
                parent = parent.parent
            
            # Only report if risky operations are not in try-except blocks
            if risky_calls and not within_try_block:
                self.add_smell('ScikitLearn', 'Exception Handling Checker', call, file_path)

    def check_imports(self, node: nodes.Module, file_path: str):
        for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
            if isinstance(import_node, nodes.Import):
                for name, alias in import_node.names:
                    if name == 'numpy' and alias != 'np':
                        self.add_smell('General', 'Import Checker', import_node, file_path)
                    elif name == 'pandas' and alias != 'pd':
                        self.add_smell('General', 'Import Checker', import_node, file_path)
            elif isinstance(import_node, nodes.ImportFrom):
                if import_node.modname == 'numpy' and any(alias != 'np' for _, alias in import_node.names):
                    self.add_smell('General', 'Import Checker', import_node, file_path)
                elif import_node.modname == 'pandas' and any(alias != 'pd' for _, alias in import_node.names):
                    self.add_smell('General', 'Import Checker', import_node, file_path)

    # PyTorch Detection Methods
    def detect_pytorch_smells(self, node: nodes.Module, file_path: str):
        """Detect PyTorch-specific code smells like:
        - Missing random seeds
        - Not using deterministic mode
        - DataLoader randomness issues
        - Missing masks for numerical operations
        - Direct forward() calls
        - Missing gradient zeroing
        - Missing batch normalization
        - Missing dropout
        - Missing data augmentation
        - Missing learning rate scheduling
        - Missing logging
        - Missing model evaluation mode

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed
        """
        self.detect_pytorch_random_seed(node, file_path)
        self.detect_pytorch_deterministic(node, file_path)
        self.detect_pytorch_dataloader_random(node, file_path)
        self.detect_pytorch_mask(node, file_path)
        self.detect_pytorch_forward(node, file_path)
        self.detect_pytorch_grad_zero(node, file_path)
        self.detect_pytorch_batch_norm(node, file_path)
        self.detect_pytorch_dropout(node, file_path)
        self.detect_pytorch_augmentation(node, file_path)
        self.detect_pytorch_lr_scheduler(node, file_path)
        self.detect_pytorch_logging(node, file_path)
        self.detect_pytorch_eval_mode(node, file_path)
 

    def detect_pytorch_random_seed(self, node: nodes.Module, file_path: str):
        # Check if there are any random operations that need seeding
        random_operations = [
            'torch.rand', 'torch.randn', 'torch.randint', 
            'torch.randperm', 'torch.bernoulli', 'torch.normal',
            'torch.dropout', 'torch.nn.Dropout'
        ]
        
        has_random_ops = any(
            op in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for op in random_operations
        )
        
        has_manual_seed = any(
            'torch.manual_seed' in call.func.as_string() or
            'torch.cuda.manual_seed' in call.func.as_string() or
            'torch.cuda.manual_seed_all' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only report if random operations are used without seeding
        if has_random_ops and not has_manual_seed:
            self.add_smell('PyTorch', 'Randomness Control Checker', node, file_path)

    def detect_pytorch_deterministic(self, node: nodes.Module, file_path: str):
        # Check for operations that benefit from deterministic algorithms
        deterministic_sensitive_ops = [
            'torch.nn.Conv', 'torch.nn.LSTM', 'torch.nn.GRU',
            'torch.backends.cudnn', 'torch.cuda', 'DataLoader'
        ]
        
        has_sensitive_ops = any(
            op in node.as_string() 
            for op in deterministic_sensitive_ops
        )
        
        has_deterministic_setting = any(
            'torch.use_deterministic_algorithms' in call.func.as_string() or
            'torch.backends.cudnn.deterministic' in call.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only report if sensitive operations are used without deterministic setting
        if has_sensitive_ops and not has_deterministic_setting:
            self.add_smell('PyTorch', 'Deterministic Algorithm Usage Checker', node, file_path)

    def detect_pytorch_dataloader_random(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'DataLoader' in call.func.as_string():
                # Check if it's actually a PyTorch DataLoader
                is_pytorch_dataloader = any(
                    'torch' in imp.names[0][0] 
                    for imp in node.nodes_of_class(nodes.ImportFrom)
                )
                
                # Check if shuffling is enabled
                has_shuffle = any(
                    (kw.arg == 'shuffle' and getattr(kw.value, 'value', False) is True)
                    for kw in call.keywords
                )
                
                # Check for proper random state control
                has_random_control = any(
                    (kw.arg in ['worker_init_fn', 'generator'])
                    for kw in call.keywords
                )
                
                # Only report if it's a PyTorch DataLoader with shuffling but no random control
                if is_pytorch_dataloader and has_shuffle and not has_random_control:
                    self.add_smell('PyTorch', 'Randomness Control Checker (PyTorch-Dataloader)', call, file_path)

    def detect_pytorch_mask(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'torch.log' in call.func.as_string():
                # Check if the input might contain zeros or negative values
                input_arg = call.args[0] if call.args else None
                if input_arg:
                    # Look for potential risky operations before the log
                    risky_ops = ['zeros', 'randn', 'rand', 'sub', 'subtract']
                    has_risky_input = any(
                        op in input_arg.as_string() 
                        for op in risky_ops
                    )
                    
                    # Check if masking is applied
                    has_mask = (
                        len(call.args) > 1 or
                        any('clamp' in node.as_string() or 
                            'mask' in node.as_string() or 
                            'where' in node.as_string())
                    )
                    
                    # Only report if there's a risk of invalid input without masking
                    if has_risky_input and not has_mask:
                        self.add_smell('PyTorch', 'Mask Missing Checker', call, file_path)

    def detect_pytorch_forward(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'forward' in call.func.as_string():
                # Check if it's a direct forward call on a neural network
                is_nn_forward = (
                    isinstance(call.func, nodes.Attribute) and
                    'forward' in call.func.attrname and
                    any(parent_cls.bases and 
                        'nn.Module' in parent_cls.bases[0].as_string()
                        for parent_cls in node.nodes_of_class(nodes.ClassDef))
                )
                
                # Check if it's called directly instead of using __call__
                is_direct_call = (
                    'net.forward' in call.func.as_string() or
                    'model.forward' in call.func.as_string()
                )
                
                # Only report if it's a direct forward call on an nn.Module
                if is_nn_forward and is_direct_call:
                    self.add_smell('PyTorch', 'Net Forward Checker', call, file_path)

    def detect_pytorch_grad_zero(self, node: nodes.Module, file_path: str):
        # Check if there's actual training happening
        has_training_loop = any(
            'loss.backward' in call.func.as_string() or
            'backward()' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_optimizer = any(
            'optim.' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_grad_zero = any(
            'zero_grad' in call.func.as_string() or
            'set_to_none' in call.func.as_string()  # Alternative method
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only report if there's training without gradient zeroing
        if has_training_loop and has_optimizer and not has_grad_zero:
            self.add_smell('PyTorch', 'Gradient Clear Checker', node, file_path)

    def detect_pytorch_batch_norm(self, node: nodes.Module, file_path: str):
        # Check if it's a CNN or deep network that could benefit from BatchNorm
        conv_layers = [
            'Conv1d', 'Conv2d', 'Conv3d',
            'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'
        ]
        
        has_conv_layers = any(
            layer in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for layer in conv_layers
        )
        
        has_deep_structure = len([
            call for call in node.nodes_of_class(nodes.Call)
            if any(layer in call.func.as_string() for layer in ['Linear', 'Conv'])
        ]) > 2
        
        has_batch_norm = any(
            'BatchNorm' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest BatchNorm for appropriate architectures
        if (has_conv_layers or has_deep_structure) and not has_batch_norm:
            self.add_smell('PyTorch', 'Batch Normalisation Checker', node, file_path)

    def detect_pytorch_dropout(self, node: nodes.Module, file_path: str):
        # Check if the model is complex enough to benefit from dropout
        has_multiple_layers = len([
            call for call in node.nodes_of_class(nodes.Call)
            if any(layer in call.func.as_string() 
                  for layer in ['Linear', 'Conv', 'LSTM', 'GRU'])
        ]) > 2
        
        has_training_code = any(
            'train()' in call.func.as_string() or
            'backward()' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_dropout = any(
            'Dropout' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest dropout for complex models during training
        if has_multiple_layers and has_training_code and not has_dropout:
            self.add_smell('PyTorch', 'Dropout Usage Checker', node, file_path)

    def detect_pytorch_augmentation(self, node: nodes.Module, file_path: str):
        # Check if it's a computer vision task
        vision_indicators = [
            'ImageFolder', 'Dataset', 'DataLoader',
            'Conv2d', 'Conv3d', 'ResNet', 'VGG',
            'image', 'img', 'PIL'
        ]
        
        is_vision_task = any(
            indicator in node.as_string() 
            for indicator in vision_indicators
        )
        
        has_training = any(
            'train()' in call.func.as_string() or
            'fit' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_augmentation = any(
            'transforms' in call.func.as_string() or
            'augment' in call.func.as_string().lower()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest augmentation for vision tasks during training
        if is_vision_task and has_training and not has_augmentation:
            self.add_smell('PyTorch', 'Data Augmentation Checker', node, file_path)

    def detect_pytorch_lr_scheduler(self, node: nodes.Module, file_path: str):
        # Check if there's a training loop with enough epochs
        has_epochs = False
        for node in node.nodes_of_class(nodes.For):
            if 'epoch' in node.as_string().lower():
                try:
                    # Try to determine the number of epochs
                    if hasattr(node.iter, 'args') and len(node.iter.args) > 0:
                        epoch_num = int(node.iter.args[0].value)
                        has_epochs = epoch_num > 10  # Suggest scheduler for longer training
                except (AttributeError, ValueError):
                    continue
        
        has_optimizer = any(
            'optim.' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_scheduler = any(
            'lr_scheduler' in call.func.as_string() or
            'LRScheduler' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest scheduler for long training processes
        if has_epochs and has_optimizer and not has_scheduler:
            self.add_smell('PyTorch', 'Learning Rate Scheduler Checker', node, file_path)

    def detect_pytorch_logging(self, node: nodes.Module, file_path: str):
        # Check if there's actual training to log
        has_training_loop = any(
            'train()' in call.func.as_string() or
            'backward()' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_metrics = any(
            metric in node.as_string().lower()
            for metric in ['loss', 'accuracy', 'score', 'metric']
        )
        
        has_logging = any(
            logger in node.as_string()
            for logger in [
                'tensorboard', 'SummaryWriter', 'wandb',
                'MLflow', 'Neptune', 'logger'
            ]
        )
        
        # Only suggest logging for training with metrics
        if has_training_loop and has_metrics and not has_logging:
            self.add_smell('PyTorch', 'Logging Checker', node, file_path)

    def detect_pytorch_eval_mode(self, node: nodes.Module, file_path: str):
        # Check if there's validation/testing happening
        evaluation_indicators = [
            'val_loader', 'test_loader', 'validate',
            'evaluation', 'testing', 'predict'
        ]
        
        has_evaluation = any(
            indicator in node.as_string().lower()
            for indicator in evaluation_indicators
        )
        
        has_model_usage = any(
            'forward' in call.func.as_string() or
            'model(' in call.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_eval_mode = any(
            'eval()' in call.func.as_string() or
            'train(False)' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest eval mode when doing validation/testing
        if has_evaluation and has_model_usage and not has_eval_mode:
            self.add_smell('PyTorch', 'Model Evaluation Checker', node, file_path)

    # TensorFlow Detection Methods
    def detect_tensorflow_smells(self, node: nodes.Module, file_path: str):
        """Detect TensorFlow-specific code smells like:
        - Missing random seeds
        - Missing early stopping
        - Missing checkpointing
        - Memory leaks
        - Missing masks
        - Python lists instead of TensorArrays
        - Missing threshold-independent metrics
        - Missing logging
        - Missing batch normalization
        - Missing dropout
        - Missing data augmentation
        - Missing learning rate scheduling
        - Missing model evaluation

        Args:
            node: AST node representing the Python module
            file_path: Path to the file being analyzed
        """
        self.detect_tf_random_seed(node, file_path)
        self.detect_tf_early_stopping(node, file_path)
        self.detect_tf_checkpointing(node, file_path)
        self.detect_tf_memory_release(node, file_path)
        self.detect_tf_mask(node, file_path)
        self.detect_tf_tensor_array(node, file_path)
        self.detect_tf_metrics(node, file_path)
        self.detect_tf_logging(node, file_path)
        self.detect_tf_batch_norm(node, file_path)
        self.detect_tf_dropout(node, file_path)
        self.detect_tf_augmentation(node, file_path)
        self.detect_tf_lr_scheduler(node, file_path)
        self.detect_tf_model_evaluation(node, file_path)


    def detect_tf_random_seed(self, node: nodes.Module, file_path: str):
        # Check for operations that need random seed control
        random_ops = [
            'random.normal', 'random.uniform', 'random.shuffle',
            'dropout', 'RandomRotation', 'RandomFlip', 'RandomZoom'
        ]
        
        has_random_ops = any(
            op in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for op in random_ops
        )
        
        has_seed_set = any(
            'random.set_seed' in call.func.as_string() or
            'set_random_seed' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only report if random operations are used without seed
        if has_random_ops and not has_seed_set:
            self.add_smell('TensorFlow', 'Randomness Control Checker', node, file_path)

    def detect_tf_early_stopping(self, node: nodes.Module, file_path: str):
        # Check if there's actual model training
        has_training = any(
            'model.fit' in call.func.as_string() or
            'fit(' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Check for training loop with multiple epochs
        has_multiple_epochs = any(
            'epochs' in kw.arg and getattr(kw.value, 'value', 1) > 1
            for call in node.nodes_of_class(nodes.Call)
            for kw in call.keywords
        )
        
        has_early_stopping = any(
            'EarlyStopping' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest early stopping for actual training with multiple epochs
        if has_training and has_multiple_epochs and not has_early_stopping:
            self.add_smell('TensorFlow', 'Early Stopping Checker', node, file_path)

    def detect_tf_checkpointing(self, node: nodes.Module, file_path: str):
        # Check for model training and complexity
        has_training = any(
            'model.fit' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_complex_model = any(
            layer in node.as_string()
            for layer in ['Dense', 'Conv', 'LSTM', 'GRU']
        )
        
        has_checkpointing = any(
            'ModelCheckpoint' in call.func.as_string() or
            'save_weights' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest checkpointing for complex models during training
        if has_training and has_complex_model and not has_checkpointing:
            self.add_smell('TensorFlow', 'Checkpointing Checker', node, file_path)

    def detect_tf_memory_release(self, node: nodes.Module, file_path: str):
        # Check for memory-intensive operations
        memory_intensive_ops = [
            'model.fit', 'predict', 'evaluate',
            'Conv', 'LSTM', 'GRU', 'Attention'
        ]
        
        has_intensive_ops = any(
            op in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
            for op in memory_intensive_ops
        )
        
        has_memory_release = any(
            'clear_session' in call.func.as_string() or
            'reset_states' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest memory release for memory-intensive operations
        if has_intensive_ops and not has_memory_release:
            self.add_smell('TensorFlow', 'Memory Release Checker', node, file_path)

    def detect_tf_mask(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if hasattr(call.func, 'as_string') and 'tf.math.log' in call.func.as_string():
                # Check for potential zero/negative inputs
                input_arg = call.args[0] if call.args else None
                if input_arg and hasattr(input_arg, 'as_string'):  # Add null check
                    risky_ops = ['zeros', 'random', 'subtract', 'sub']
                    has_risky_input = any(
                        op in input_arg.as_string()
                        for op in risky_ops
                    )
                    
                    has_mask = (
                        len(call.args) > 1 or
                        any(mask_op in node.as_string() 
                            for mask_op in ['where', 'clip', 'maximum'])
                    )
                    
                    # Only report if there's risk of invalid input without masking
                    if has_risky_input and not has_mask:
                        self.add_smell('TensorFlow', 'Mask Missing Checker', call, file_path)

    def detect_tf_tensor_array(self, node: nodes.Module, file_path: str):
        # Check for dynamic sequence operations
        dynamic_ops = [
            'RNN', 'LSTM', 'GRU', 'while_loop',
            'map_fn', 'scan', 'dynamic'
        ]
        
        has_dynamic_ops = any(
            op in node.as_string()
            for op in dynamic_ops
        )
        
        using_python_list = any(
            'append' in call.func.as_string() or
            'extend' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_tensor_array = any(
            'TensorArray' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest TensorArray for dynamic operations using Python lists
        if has_dynamic_ops and using_python_list and not has_tensor_array:
            self.add_smell('TensorFlow', 'Tensor Array Checker', node, file_path)

    def detect_tf_metrics(self, node: nodes.Module, file_path: str):
        # Check if it's a classification task
        classification_indicators = [
            'Binary', 'Categorical', 'sparse_categorical',
            'accuracy', 'precision', 'recall', 'f1'
        ]
        
        is_classification = any(
            indicator in node.as_string()
            for indicator in classification_indicators
        )
        
        has_basic_metrics = any(
            metric in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
            for metric in ['accuracy', 'precision', 'recall']
        )
        
        has_threshold_independent = any(
            metric in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
            for metric in ['AUC', 'AUROCScore', 'PrecisionRecallCurve']
        )
        
        # Only suggest threshold-independent metrics for classification tasks
        if is_classification and has_basic_metrics and not has_threshold_independent:
            self.add_smell('TensorFlow', 'Dependent Threshold Checker', node, file_path)

    def detect_tf_logging(self, node: nodes.Module, file_path: str):
        # Check if there's actual training to log
        has_training_loop = any(
            'model.fit' in call.func.as_string() or
            'train' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_metrics = any(
            metric in node.as_string().lower()
            for metric in ['loss', 'accuracy', 'score', 'metric']
        )
        
        has_logging = any(
            logger in node.as_string()
            for logger in [
                'tf.summary', 'TensorBoard', 'CSVLogger',
                'WandbCallback', 'MLflow'
            ]
        )
        
        # Only suggest logging for training with metrics
        if has_training_loop and has_metrics and not has_logging:
            self.add_smell('TensorFlow', 'Logging Checker', node, file_path)

    def detect_tf_batch_norm(self, node: nodes.Module, file_path: str):
        # Check if it's a deep network that could benefit from BatchNorm
        conv_layers = [
            'Conv1D', 'Conv2D', 'Conv3D',
            'Dense', 'SeparableConv'
        ]
        
        has_conv_layers = any(
            layer in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for layer in conv_layers
        )
        
        # Check if model is deep enough
        layer_count = len([
            call for call in node.nodes_of_class(nodes.Call)
            if any(layer in call.func.as_string() 
                  for layer in conv_layers)
        ])
        
        has_batch_norm = any(
            'BatchNormalization' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest BatchNorm for deep networks with conv layers
        if has_conv_layers and layer_count > 2 and not has_batch_norm:
            self.add_smell('TensorFlow', 'Batch Normalisation Checker', node, file_path)

    def detect_tf_dropout(self, node: nodes.Module, file_path: str):
        # Check if model is complex enough to need dropout
        deep_layers = ['Dense', 'Conv', 'LSTM', 'GRU']
        
        layer_count = len([
            call for call in node.nodes_of_class(nodes.Call)
            if any(layer in call.func.as_string() 
                  for layer in deep_layers)
        ])
        
        has_training = any(
            'model.fit' in call.func.as_string() or
            'training=True' in call.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_dropout = any(
            'Dropout' in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest dropout for complex models during training
        if layer_count > 2 and has_training and not has_dropout:
            self.add_smell('TensorFlow', 'Dropout Usage Checker', node, file_path)

    def detect_tf_augmentation(self, node: nodes.Module, file_path: str):
        # Check if it's a computer vision task
        vision_indicators = [
            'image', 'img', 'Conv2D', 'Conv3D',
            'ImageDataGenerator', 'load_img'
        ]
        
        is_vision_task = any(
            indicator in node.as_string() 
            for indicator in vision_indicators
        )
        
        has_training = any(
            'model.fit' in call.func.as_string() or
            'train' in call.func.as_string().lower()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_augmentation = any(
            aug in call.func.as_string() 
            for call in node.nodes_of_class(nodes.Call)
            for aug in ['ImageDataGenerator', 'RandomFlip', 'RandomRotation', 'RandomZoom']
        )
        
        # Only suggest augmentation for vision tasks during training
        if is_vision_task and has_training and not has_augmentation:
            self.add_smell('TensorFlow', 'Data Augmentation Checker', node, file_path)

    def detect_tf_lr_scheduler(self, node: nodes.Module, file_path: str):
        # Check if there's a long training process
        has_epochs = False
        for node in node.nodes_of_class(nodes.For):
            if 'epoch' in node.as_string().lower():
                try:
                    if hasattr(node.iter, 'args') and len(node.iter.args) > 0:
                        epoch_num = int(node.iter.args[0].value)
                        has_epochs = epoch_num > 5  # Suggest scheduler for longer training
                except (AttributeError, ValueError):
                    continue
        
        has_optimizer = any(
            'optimizer' in call.func.as_string().lower()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_scheduler = any(
            'LearningRateScheduler' in call.func.as_string() or
            'schedules' in call.func.as_string() or
            'ReduceLROnPlateau' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest scheduler for long training processes
        if has_epochs and has_optimizer and not has_scheduler:
            self.add_smell('TensorFlow', 'Learning Rate Scheduler Checker', node, file_path)

    def detect_tf_model_evaluation(self, node: nodes.Module, file_path: str):
        # Check if there's a model to evaluate
        has_model = any(
            'model' in call.func.as_string() or
            'Sequential' in call.func.as_string() or
            'Model(' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        has_test_data = any(
            data in node.as_string().lower()
            for data in ['test', 'val', 'valid', 'evaluation']
        )
        
        has_evaluation = any(
            'evaluate' in call.func.as_string() or
            'predict' in call.func.as_string()
            for call in node.nodes_of_class(nodes.Call)
        )
        
        # Only suggest evaluation when there's a model and test data
        if has_model and has_test_data and not has_evaluation:
            self.add_smell('TensorFlow', 'Model Evaluation Checker', node, file_path)


    def generate_report(self) -> str:
        """Generate a detailed text report of all detected code smells.

        Returns:
            A formatted string containing the analysis report with:
            - Framework-specific smell counts
            - Details for each smell including location and fixes
            - Total number of smells detected
        """
        report = "Framework-Specific Code Smell Report\n====================================\n\n"
        smell_counts = {}
        for smell in self.smells:
            framework = smell['framework']
            if framework not in smell_counts:
                smell_counts[framework] = {}
            if smell['name'] not in smell_counts[framework]:
                smell_counts[framework][smell['name']] = 0
            smell_counts[framework][smell['name']] += 1

            report += f"Framework: {framework}\n"
            report += f"Smell: {smell['name']}\n"
            report += f"File: {smell['file_path']}\n"
            
            if smell['line_number'] != 0:
                report += f"Line: {smell['line_number']}\n"
            
            code_lines = smell['code_snippet'].strip().split('\n')
            if len(code_lines) <= 3:
                report += f"Code Snippet:\n{smell['code_snippet']}\n"
            
            report += f"How to Fix: {smell['how_to_fix']}\n"
            report += f"Benefits: {smell['benefits']}\n"
            report += f"Strategies: {smell['strategies']}\n\n"

        report += "Smell Counts:\n"
        for framework, counts in smell_counts.items():
            report += f"{framework}:\n"
            for smell, count in counts.items():
                report += f"  {smell}: {count}\n"
        report += f"\nTotal smells detected: {len(self.smells)}"
        return report

    def get_results(self) -> List[Dict[str, str]]:
        """Get a simplified list of detected code smells.

        Returns:
            List of dictionaries containing:
            - framework: The ML framework
            - name: Name of the smell
            - fix: How to fix it
            - benefits: Benefits of fixing
            - location: Where it was found
        """
        return [
            {
                'framework': smell['framework'],
                'name': smell['name'],
                'fix': smell['how_to_fix'],
                'benefits': smell['benefits'],
                'location': f"Line {smell['line_number']}" if smell['line_number'] != 0 else ""
            }
            for smell in self.smells
        ]

    def get_smells(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "General": [
                {
                    "name": "Import Checker",
                    "how_to_fix": "Use standard naming conventions for imported modules.",
                    "benefits": "Improves code readability and maintainability.",
                    "strategies": "Follow standard naming conventions (e.g., import numpy as np, import pandas as pd)."
                }
            ],
            "Pandas": [
                {
                    "name": "Unnecessary Iteration",
                    "how_to_fix": "Use vectorized operations instead of loops.",
                    "benefits": "Enhances performance and reduces execution time.",
                    "strategies": "Replace loops with Pandas vectorized functions (e.g., apply, map, vectorized arithmetic operations)."
                },
                {
                    "name": "DataFrame Iteration Modification",
                    "how_to_fix": "Avoid modifying DataFrame during iteration.",
                    "benefits": "Prevents unexpected behaviour and potential data corruption.",
                    "strategies": "Use temporary variables or vectorized operations for modifications."
                },
                {
                    "name": "Chain Indexing",
                    "how_to_fix": "Use single indexing or .loc[], .iloc[] methods.",
                    "benefits": "Enhances code readability and prevents performance issues.",
                    "strategies": "Use .loc[] or .iloc[] for DataFrame indexing instead of chained indexing."
                },
                {
                    "name": "Datatype Checker",
                    "how_to_fix": "Set data types explicitly when importing data.",
                    "benefits": "Ensures correct data format and reduces memory usage.",
                    "strategies": "Use dtype parameter in Pandas read functions (e.g., pd.read_csv)."
                },
                {
                    "name": "Column Selection Checker",
                    "how_to_fix": "Select necessary columns after importing DataFrame.",
                    "benefits": "Clarifies data usage and improves performance.",
                    "strategies": "Use column selection methods (e.g., df[['col1', 'col2']])."
                },
                {
                    "name": "Merge Parameter Checker",
                    "how_to_fix": "Specify how, on, and validate parameters in merge operations.",
                    "benefits": "Ensures accurate data merging and prevents data loss.",
                    "strategies": "Use appropriate parameters in pd.merge function."
                },
                {
                    "name": "InPlace Checker",
                    "how_to_fix": "Assign operations to a new DataFrame variable.",
                    "benefits": "Prevents data loss and improves code clarity.",
                    "strategies": "Assign operation results to a new variable instead of using inplace=True."
                },
                {
                    "name": "DataFrame Conversion Checker",
                    "how_to_fix": "Use .to_numpy() instead of .values.",
                    "benefits": "Ensures future compatibility and avoids unexpected behaviour.",
                    "strategies": "Replace .values with .to_numpy() in Pandas DataFrame conversion."
                }
            ],
            "NumPy": [
                {
                    "name": "NaN Equality Checker",
                    "how_to_fix": "Use np.isnan() to check for NaN values.",
                    "benefits": "Ensures accurate data handling and avoids logical errors.",
                    "strategies": "Replace == np.nan with np.isnan()."
                },
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Use np.random.seed() for reproducibility.",
                    "benefits": "Enables reproducible results and debugging.",
                    "strategies": "Set random seed using np.random.seed(seed_value)."
                },
                {
                    "name": "Array Creation Efficiency",
                    "how_to_fix": "Specify dtype when creating arrays.",
                    "benefits": "Improves memory efficiency and avoids unnecessary conversions.",
                    "strategies": "Use dtype parameter in np.array, np.zeros, np.ones, np.empty functions."
                },
                {
                    "name": "Inefficient Operations",
                    "how_to_fix": "Optimize numerical operations.",
                    "benefits": "Improves performance and reduces execution time.",
                    "strategies": "Use vectorized functions instead of loops for element-wise operations."
                },
                {
                    "name": "Dtype Consistency",
                    "how_to_fix": "Ensure consistent data types in operations.",
                    "benefits": "Improves accuracy and avoids logical errors.",
                    "strategies": "Use consistent data types in np.sum, np.mean, np.max, np.min functions."
                },
                {
                    "name": "Broadcasting Risk",
                    "how_to_fix": "Specify axis in operations between arrays.",
                    "benefits": "Improves performance and avoids broadcasting issues.",
                    "strategies": "Use axis parameter in np.reshape, np.transpose functions."
                },
                {
                    "name": "Copy-View Confusion",
                    "how_to_fix": "Explicitly copy arrays before slicing.",
                    "benefits": "Improves performance and avoids unexpected behaviour.",
                    "strategies": "Use .copy() method or np.copy function before slicing."
                },
                {
                    "name": "Missing Axis Specification",
                    "how_to_fix": "Specify axis in array operations.",
                    "benefits": "Improves performance and avoids unexpected behaviour.",
                    "strategies": "Use axis parameter in np.sum, np.mean, np.max, np.min, np.argmax, np.argmin, np.any, np.all functions."
                }
            ],
            "ScikitLearn": [
                {
                    "name": "Scaler Missing Checker",
                    "how_to_fix": "Apply scaling before scaling-sensitive operations.",
                    "benefits": "Improves model performance and accuracy.",
                    "strategies": "Use StandardScaler, MinMaxScaler, etc., before applying PCA, SVM, etc."
                },
                {
                    "name": "Pipeline Checker",
                    "how_to_fix": "Use Pipelines for all scikit-learn estimators.",
                    "benefits": "Prevents data leakage and ensures correct model evaluation.",
                    "strategies": "Implement Pipeline from sklearn.pipeline for preprocessing and model fitting."
                },
                {
                    "name": "Cross Validation Checker",
                    "how_to_fix": "Implement cross-validation techniques for robust model evaluation.",
                    "benefits": "Enhances model performance and reduces overfitting.",
                    "strategies": "Use cross_val_score, KFold, or other cross-validation methods."
                },
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Set random_state in estimators for reproducibility.",
                    "benefits": "Ensures reproducible results and consistent model behaviour.",
                    "strategies": "Set random_state to a fixed value in scikit-learn estimators."
                },
                {
                    "name": "Verbose Mode Checker",
                    "how_to_fix": "Enable verbose mode for long training processes.",
                    "benefits": "Provides better insights into model training progress.",
                    "strategies": "Set verbose=True in scikit-learn estimators with long training times."
                },
                {
                    "name": "Dependent Threshold Checker",
                    "how_to_fix": "Use threshold-independent metrics alongside threshold-dependent ones.",
                    "benefits": "Provides a comprehensive evaluation of model performance.",
                    "strategies": "Include metrics like ROC AUC score alongside accuracy or F1-score."
                },
                {
                    "name": "Unit Testing Checker",
                    "how_to_fix": "Write unit tests for data processing and model components.",
                    "benefits": "Ensures code reliability and prevents bugs.",
                    "strategies": "Use unittest or pytest to write and run tests for individual components."
                },
                {
                    "name": "Data Leakage Checker",
                    "how_to_fix": "Split data before any preprocessing or feature engineering.",
                    "benefits": "Prevents data leakage and ensures valid model evaluation.",
                    "strategies": "Use train_test_split before any data preprocessing steps."
                },
                {
                    "name": "Exception Handling Checker",
                    "how_to_fix": "Implement proper exception handling for model operations.",
                    "benefits": "Improves code robustness and error reporting.",
                    "strategies": "Use try-except blocks to handle potential exceptions in model training and prediction."
                }
            ],
            "PyTorch": [
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Set random seed using torch.manual_seed() for reproducibility.",
                    "benefits": "Ensures reproducible results across different runs.",
                    "strategies": "Add torch.manual_seed(seed_value) at the start of your script."
                },
                {
                    "name": "Deterministic Algorithm Usage Checker",
                    "how_to_fix": "Enable deterministic algorithms using torch.use_deterministic_algorithms(True).",
                    "benefits": "Ensures consistent results across different hardware and runs.",
                    "strategies": "Set torch.use_deterministic_algorithms(True) and handle any required adjustments."
                },
                {
                    "name": "Randomness Control Checker (PyTorch-Dataloader)",
                    "how_to_fix": "Set worker_init_fn and generator in DataLoader for reproducible data loading.",
                    "benefits": "Ensures consistent data loading across different runs.",
                    "strategies": "Configure worker_init_fn and generator parameters in DataLoader initialization."
                },
                {
                    "name": "Mask Missing Checker",
                    "how_to_fix": "Use appropriate masking when dealing with log operations.",
                    "benefits": "Prevents numerical errors and improves model stability.",
                    "strategies": "Apply masks before log operations to handle invalid values."
                },
                {
                    "name": "Net Forward Checker",
                    "how_to_fix": "Use model(input) instead of model.forward(input).",
                    "benefits": "Follows PyTorch best practices and handles hooks properly.",
                    "strategies": "Replace direct forward() calls with the recommended calling syntax."
                },
                {
                    "name": "Gradient Clear Checker",
                    "how_to_fix": "Clear gradients before each backward pass using optimizer.zero_grad().",
                    "benefits": "Prevents gradient accumulation and ensures correct updates.",
                    "strategies": "Add optimizer.zero_grad() before loss.backward() in training loop."
                },
                {
                    "name": "Batch Normalisation Checker",
                    "how_to_fix": "Include BatchNorm layers in your model architecture.",
                    "benefits": "Improves training stability and model convergence.",
                    "strategies": "Add torch.nn.BatchNorm layers after convolutional or linear layers."
                },
                {
                    "name": "Dropout Usage Checker",
                    "how_to_fix": "Include Dropout layers for regularization.",
                    "benefits": "Reduces overfitting and improves model generalization.",
                    "strategies": "Add torch.nn.Dropout layers in your model architecture."
                },
                {
                    "name": "Data Augmentation Checker",
                    "how_to_fix": "Implement data augmentation using torchvision.transforms.",
                    "benefits": "Improves model robustness and reduces overfitting.",
                    "strategies": "Use torchvision.transforms to apply various augmentation techniques."
                },
                {
                    "name": "Learning Rate Scheduler Checker",
                    "how_to_fix": "Implement learning rate scheduling.",
                    "benefits": "Improves training convergence and final model performance.",
                    "strategies": "Use torch.optim.lr_scheduler for dynamic learning rate adjustment."
                },
                {
                    "name": "Logging Checker",
                    "how_to_fix": "Implement proper logging using tensorboard or similar tools.",
                    "benefits": "Enables better experiment tracking and debugging.",
                    "strategies": "Use tensorboardX or torch.utils.tensorboard for logging."
                },
                {
                    "name": "Model Evaluation Checker",
                    "how_to_fix": "Set model to evaluation mode using model.eval().",
                    "benefits": "Ensures correct behavior of layers like BatchNorm and Dropout during inference.",
                    "strategies": "Call model.eval() before validation/testing and model.train() before training."
                }
            ],
            "TensorFlow": [
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Set random seed using tf.random.set_seed().",
                    "benefits": "Ensures reproducible results across different runs.",
                    "strategies": "Add tf.random.set_seed(seed_value) at the start of your script."
                },
                {
                    "name": "Early Stopping Checker",
                    "how_to_fix": "Implement early stopping using tf.keras.callbacks.EarlyStopping.",
                    "benefits": "Prevents overfitting and reduces unnecessary training time.",
                    "strategies": "Add EarlyStopping callback to model.fit()."
                },
                {
                    "name": "Checkpointing Checker",
                    "how_to_fix": "Implement model checkpointing using tf.keras.callbacks.ModelCheckpoint.",
                    "benefits": "Enables model recovery and saves best performing models.",
                    "strategies": "Add ModelCheckpoint callback to model.fit()."
                },
                {
                    "name": "Memory Release Checker",
                    "how_to_fix": "Clear session after model creation/training using tf.keras.backend.clear_session().",
                    "benefits": "Prevents memory leaks and reduces resource usage.",
                    "strategies": "Call clear_session() after completing major operations."
                },
                {
                    "name": "Mask Missing Checker",
                    "how_to_fix": "Use appropriate masking for log operations.",
                    "benefits": "Prevents numerical errors and improves model stability.",
                    "strategies": "Apply masks before log operations to handle invalid values."
                },
                {
                    "name": "Tensor Array Checker",
                    "how_to_fix": "Use tf.TensorArray for dynamic tensor operations.",
                    "benefits": "Improves memory efficiency for dynamic computations.",
                    "strategies": "Replace Python lists with tf.TensorArray for growing tensors."
                },
                {
                    "name": "Dependent Threshold Checker",
                    "how_to_fix": "Use threshold-independent metrics like AUC.",
                    "benefits": "Provides more robust model evaluation.",
                    "strategies": "Include tf.keras.metrics.AUC in your model metrics."
                },
                {
                    "name": "Logging Checker",
                    "how_to_fix": "Implement TensorBoard logging.",
                    "benefits": "Enables better experiment tracking and visualization.",
                    "strategies": "Use tf.summary or TensorBoard callbacks for logging."
                },
                {
                    "name": "Batch Normalisation Checker",
                    "how_to_fix": "Include BatchNormalization layers in your model.",
                    "benefits": "Improves training stability and model convergence.",
                    "strategies": "Add tf.keras.layers.BatchNormalization layers to your model."
                },
                {
                    "name": "Dropout Usage Checker",
                    "how_to_fix": "Include Dropout layers for regularization.",
                    "benefits": "Reduces overfitting and improves generalization.",
                    "strategies": "Add tf.keras.layers.Dropout layers to your model."
                },
                {
                    "name": "Data Augmentation Checker",
                    "how_to_fix": "Implement data augmentation using ImageDataGenerator.",
                    "benefits": "Improves model robustness and reduces overfitting.",
                    "strategies": "Use tf.keras.preprocessing.image.ImageDataGenerator for augmentation."
                },
                {
                    "name": "Learning Rate Scheduler Checker",
                    "how_to_fix": "Implement learning rate scheduling.",
                    "benefits": "Improves training convergence and final model performance.",
                    "strategies": "Use LearningRateScheduler callback or custom scheduling."
                },
                {
                    "name": "Model Evaluation Checker",
                    "how_to_fix": "Evaluate model performance using model.evaluate().",
                    "benefits": "Provides standardized model evaluation.",
                    "strategies": "Use model.evaluate() on validation/test data."
                },
                {
                    "name": "Unit Testing Checker",
                    "how_to_fix": "Implement unit tests using tf.test.TestCase.",
                    "benefits": "Ensures code reliability and prevents regressions.",
                    "strategies": "Create test classes inheriting from tf.test.TestCase."
                },
                {
                    "name": "Exception Handling Checker",
                    "how_to_fix": "Implement proper exception handling.",
                    "benefits": "Improves code robustness and error reporting.",
                    "strategies": "Use try-except blocks to handle TensorFlow-specific exceptions."
                }
            ]
        }

