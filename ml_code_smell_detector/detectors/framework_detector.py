import astroid
from astroid import nodes
from typing import List, Dict, Any

class FrameworkSpecificSmellDetector:
    def __init__(self):
        self.smells: List[Dict[str, Any]] = []
        self.framework_smells = self.get_smells()

    def detect_smells(self, file_path: str) -> List[Dict[str, str]]:
        with open(file_path, 'r') as file:
            content = file.read()
        module = astroid.parse(content, module_name=file_path)
        self.visit_module(module, file_path)
        return self.smells

    def visit_module(self, node: nodes.Module, file_path: str):
        self.check_imports(node, file_path)
        self.check_pandas_smells(node, file_path)
        self.check_numpy_smells(node, file_path)
        self.check_sklearn_smells(node, file_path)
        self.check_tensorflow_smells(node, file_path)
        self.check_pytorch_smells(node, file_path)

    def add_smell(self, framework: str, smell_name: str, node: nodes.NodeNG, file_path: str):
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

    def check_pandas_smells(self, node: nodes.Module, file_path: str):
        for call in node.nodes_of_class(nodes.Call):
            if 'iterrows' in call.func.as_string():
                self.add_smell('Pandas', 'Unnecessary Iteration', call, file_path)
            if 'merge' in call.func.as_string() and not any(kw.arg in ['how', 'on', 'validate'] for kw in call.keywords):
                self.add_smell('Pandas', 'Merge Parameter Checker', call, file_path)
            if 'inplace=True' in call.as_string():
                self.add_smell('Pandas', 'InPlace Checker', call, file_path)
            if '.values' in call.as_string():
                self.add_smell('Pandas', 'DataFrame Conversion Checker', call, file_path)
            if 'read_csv' in call.func.as_string() and not any(kw.arg == 'dtype' for kw in call.keywords):
                self.add_smell('Pandas', 'Datatype Checker', call, file_path)

        for subscript in node.nodes_of_class(nodes.Subscript):
            if isinstance(subscript.value, nodes.Subscript):
                self.add_smell('Pandas', 'Chain Indexing', subscript, file_path)

        for assign in node.nodes_of_class(nodes.Assign):
            if isinstance(assign.targets[0], nodes.Subscript) and isinstance(assign.targets[0].value, nodes.Name):
                self.add_smell('Pandas', 'DataFrame Iteration Modification', assign, file_path)

        if not any('[[' in subscript.as_string() for subscript in node.nodes_of_class(nodes.Subscript)):
            self.add_smell('Pandas', 'Column Selection Checker', node, file_path)



    def check_numpy_smells(self, node: nodes.Module, file_path: str):
        for compare in node.nodes_of_class(nodes.Compare):
            if any('np.nan' in getattr(op[0], 'as_string', lambda: '')() for op in compare.ops):
                self.add_smell('NumPy', 'NaN Equality Checker', compare, file_path)

        random_seed_calls = [call for call in node.nodes_of_class(nodes.Call) if 'np.random.seed' in getattr(call.func, 'as_string', lambda: '')()]
        if not random_seed_calls:
            self.add_smell('NumPy', 'Randomness Control Checker', node, file_path)

    def check_sklearn_smells(self, node: nodes.Module, file_path: str):
        scaling_methods = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
        if any(method in call.func.as_string() for call in node.nodes_of_class(nodes.Call) for method in scaling_methods):
            self.add_smell('ScikitLearn', 'Scaler Missing Checker', node, file_path)

        if not any('Pipeline' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('ScikitLearn', 'Pipeline Checker', node, file_path)

        if not any('cross_val_score' in call.func.as_string() or 'KFold' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('ScikitLearn', 'Cross Validation Checker', node, file_path)

        if not any('random_state' in kw.arg for call in node.nodes_of_class(nodes.Call) for kw in call.keywords):
            self.add_smell('ScikitLearn', 'Randomness Control Checker', node, file_path)

        if not any('verbose=True' in call.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('ScikitLearn', 'Verbose Mode Checker', node, file_path)

        if not any('roc_auc_score' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('ScikitLearn', 'Dependent Threshold Checker', node, file_path)

        if not any('unittest' in imp.names[0][0] or 'pytest' in imp.names[0][0] for imp in node.nodes_of_class(nodes.ImportFrom)):
            self.add_smell('ScikitLearn', 'Unit Testing Checker', node, file_path)

        if not any('train_test_split' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('ScikitLearn', 'Data Leakage Checker', node, file_path)

        if not any(isinstance(n, nodes.Try) for n in node.body):
            self.add_smell('ScikitLearn', 'Exception Handling Checker', node, file_path)

    def check_tensorflow_smells(self, node: nodes.Module, file_path: str):
        if not any('tf.random.set_seed' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Randomness Control Checker', node, file_path)

        if not any('tf.keras.callbacks.EarlyStopping' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Early Stopping Checker', node, file_path)

        if not any('tf.keras.callbacks.ModelCheckpoint' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Checkpointing Checker', node, file_path)

        if not any('tf.keras.backend.clear_session' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Memory Release Checker', node, file_path)

        if not any('tf.math.log' in call.func.as_string() and len(call.args) > 1 for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Mask Missing Checker', node, file_path)

        if not any('tf.TensorArray' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Tensor Array Checker', node, file_path)

        if not any('tf.keras.metrics.AUC' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Dependent Threshold Checker', node, file_path)

        if not any('tf.summary' in call.func.as_string() or 'TensorBoard' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Logging Checker', node, file_path)

        if not any('tf.keras.layers.BatchNormalization' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Batch Normalisation Checker', node, file_path)

        if not any('tf.keras.layers.Dropout' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Dropout Usage Checker', node, file_path)

        if not any('tf.keras.preprocessing.image.ImageDataGenerator' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Data Augmentation Checker', node, file_path)

        if not any('tf.keras.callbacks.LearningRateScheduler' in call.func.as_string() or 'tf.keras.optimizers.schedules' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Learning Rate Scheduler Checker', node, file_path)

        if not any('model.evaluate' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('TensorFlow', 'Model Evaluation Checker', node, file_path)

        if not any('tf.test.TestCase' in base.as_string() for cls in node.nodes_of_class(nodes.ClassDef) for base in cls.bases):
            self.add_smell('TensorFlow', 'Unit Testing Checker', node, file_path)

        if not any(isinstance(n, nodes.Try) for n in node.body):
            self.add_smell('TensorFlow', 'Exception Handling Checker', node, file_path)

    def check_pytorch_smells(self, node: nodes.Module, file_path: str):
        if not any('torch.manual_seed' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Randomness Control Checker', node, file_path)

        if not any('torch.use_deterministic_algorithms' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Deterministic Algorithm Usage Checker', node, file_path)

        if not any('DataLoader' in call.func.as_string() and 'worker_init_fn' in call.as_string() and 'generator' in call.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Randomness Control Checker (PyTorch-Dataloader)', node, file_path)

        if not any('torch.log' in call.func.as_string() and len(call.args) > 1 for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Mask Missing Checker', node, file_path)

        if any('net.forward' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Net Forward Checker', node, file_path)

        if not any('optimizer.zero_grad' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Gradient Clear Checker', node, file_path)

        if not any('torch.nn.BatchNorm' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Batch Normalisation Checker', node, file_path)

        if not any('torch.nn.Dropout' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Dropout Usage Checker', node, file_path)

        if not any('torchvision.transforms' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Data Augmentation Checker', node, file_path)

        if not any('torch.optim.lr_scheduler' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Learning Rate Scheduler Checker', node, file_path)

        if not any('tensorboardX' in imp.names[0][0] or 'SummaryWriter' in imp.names[0][0] for imp in node.nodes_of_class(nodes.ImportFrom)):
            self.add_smell('PyTorch', 'Logging Checker', node, file_path)

        if not any('model.eval()' in call.func.as_string() for call in node.nodes_of_class(nodes.Call)):
            self.add_smell('PyTorch', 'Model Evaluation Checker', node, file_path)

        if not any('unittest' in imp.names[0][0] or 'pytest' in imp.names[0][0] for imp in node.nodes_of_class(nodes.ImportFrom)):
            self.add_smell('PyTorch', 'Unit Testing Checker', node, file_path)

        if not any(isinstance(n, nodes.Try) for n in node.body):
            self.add_smell('PyTorch', 'Exception Handling Checker', node, file_path)

    def get_smells(self) -> Dict[str, List[Dict[str, str]]]:
        return {
            "General": [
                {
                    "name": "Import Checker",
                    "how_to_fix": "Use standard naming conventions for imported modules.",
                    "benefits": "Improves code readability and maintainability.",
                    "strategies": "Follow standard naming conventions (e.g., import numpy as np)."
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
                    "how_to_fix": "Use np.seed() for reproducibility.",
                    "benefits": "Enables reproducible results and debugging.",
                    "strategies": "Set random seed using np.random.seed(seed_value)."
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
                    "how_to_fix": "Remove random_state=None in estimators.",
                    "benefits": "Ensures reproducible results and consistent model behaviour.",
                    "strategies": "Set random_state to a fixed value in scikit-learn estimators."
                },
                {
                    "name": "Verbose Mode Checker",
                    "how_to_fix": "Use verbose mode for long training processes.",
                    "benefits": "Provides better insights into model training progress and potential issues.",
                    "strategies": "Set verbose=True in scikit-learn estimators with long training times."
                },
                {
                    "name": "Dependent Threshold Checker",
                    "how_to_fix": "Use threshold-independent metrics alongside threshold-dependent metrics.",
                    "benefits": "Provides a comprehensive evaluation of model performance.",
                    "strategies": "Include metrics like AUC with f-score for evaluation."
                },
                {
                    "name": "Unit Testing Checker",
                    "how_to_fix": "Write unit tests for data processing and model components.",
                    "benefits": "Ensures code reliability and prevents bugs.",
                    "strategies": "Use unittest or pytest to write and run tests for individual components."
                },
                {
                    "name": "Data Leakage Checker",
                    "how_to_fix": "Ensure no data leakage between training and test sets.",
                    "benefits": "Maintains model integrity and accurate performance metrics.",
                    "strategies": "Use cross-validation techniques and separate preprocessing for training and testing."
                },
                {
                    "name": "Exception Handling Checker",
                    "how_to_fix": "Handle exceptions in data processing and model training steps.",
                    "benefits": "Prevents crashes and provides informative error messages.",
                    "strategies": "Use try-except blocks to handle exceptions during data processing and model training in ScikitLearn."
                }
            ],
            "TensorFlow": [
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Use tf.random.set_seed() for reproducibility.",
                    "benefits": "Ensures reproducible training results.",
                    "strategies": "Set TensorFlow random seed using tf.random.set_seed(seed_value)."
                },
                {
                    "name": "Early Stopping Checker",
                    "how_to_fix": "Implement early stopping to prevent overfitting.",
                    "benefits": "Reduces overfitting and training time, improves generalisation.",
                    "strategies": "Use tf.keras.callbacks.EarlyStopping in TensorFlow training process."
                },
                {
                    "name": "Checkpointing Checker",
                    "how_to_fix": "Save model checkpoints during training.",
                    "benefits": "Prevents data loss and allows training to resume from a specific point.",
                    "strategies": "Use tf.keras.callbacks.ModelCheckpoint for checkpointing during training."
                },
                {
                    "name": "Memory Release Checker",
                    "how_to_fix": "Clear memory if a neural network is created in a loop.",
                    "benefits": "Prevents memory leaks and improves performance.",
                    "strategies": "Use tf.keras.backend.clear_session() after model creation."
                },
                {
                    "name": "Mask Missing Checker",
                    "how_to_fix": "Ensure valid arguments for log functions.",
                    "benefits": "Prevents numerical errors and improves model stability.",
                    "strategies": "Validate arguments before using log functions."
                },
                {
                    "name": "Tensor Array Checker",
                    "how_to_fix": "Use tf.TensorArray() for growing arrays in loops.",
                    "benefits": "Enhances performance and reduces memory usage.",
                    "strategies": "Replace growing lists with tf.TensorArray()."
                },
                {
                    "name": "Dependent Threshold Checker",
                    "how_to_fix": "Use threshold-independent metrics alongside threshold-dependent metrics.",
                    "benefits": "Provides a comprehensive evaluation of model performance.",
                    "strategies": "Include metrics like AUC with f-score for evaluation."
                },
                {
                    "name": "Logging Checker",
                    "how_to_fix": "Use logging for tracking experiments and results.",
                    "benefits": "Facilitates debugging and experiment tracking, improves reproducibility.",
                    "strategies": "Implement logging using TensorFlow's tf.summary or external libraries like TensorBoard."
                },
                {
                    "name": "Batch Normalisation Checker",
                    "how_to_fix": "Use batch normalisation layers to improve training stability.",
                    "benefits": "Stabilises and accelerates training, improves model performance.",
                    "strategies": "Add tf.keras.layers.BatchNormalization to the model architecture."
                },
                {
                    "name": "Dropout Usage Checker",
                    "how_to_fix": "Apply dropout layers to reduce overfitting.",
                    "benefits": "Reduces overfitting and improves model generalisation.",
                    "strategies": "Include tf.keras.layers.Dropout in the model architecture."
                },
                {
                    "name": "Data Augmentation Checker",
                    "how_to_fix": "Apply data augmentation techniques to enhance model robustness.",
                    "benefits": "Improves model generalisation and robustness.",
                    "strategies": "Use tf.keras.preprocessing.image.ImageDataGenerator for data augmentation."
                },
                {
                    "name": "Learning Rate Scheduler Checker",
                    "how_to_fix": "Implement learning rate schedulers for dynamic learning rate adjustment.",
                    "benefits": "Optimises training process and improves model performance.",
                    "strategies": "Use tf.keras.callbacks.LearningRateScheduler or tf.keras.optimizers.schedules for learning rate scheduling."
                },
                {
                    "name": "Model Evaluation Checker",
                    "how_to_fix": "Evaluate model performance on validation/test data regularly.",
                    "benefits": "Ensures model is generalising well to unseen data.",
                    "strategies": "Regularly validate and test model during and after training."
                },
                {
                    "name": "Unit Testing Checker",
                    "how_to_fix": "Develop unit tests for TensorFlow components and workflows.",
                    "benefits": "Enhances code robustness and reliability.",
                    "strategies": "Use tf.test.TestCase or external testing frameworks for unit testing TensorFlow code."
                },
                {
                    "name": "Exception Handling Checker",
                    "how_to_fix": "Implement proper exception handling for model training and inference.",
                    "benefits": "Improves code robustness and error management.",
                    "strategies": "Use try-except blocks to handle potential exceptions during training and inference."
                }
            ],
            "PyTorch": [
                {
                    "name": "Randomness Control Checker",
                    "how_to_fix": "Use torch.manual_seed() for reproducibility.",
                    "benefits": "Enables consistent and reproducible results.",
                    "strategies": "Set random seed in PyTorch using torch.manual_seed(seed_value)."
                },
                {
                    "name": "Deterministic Algorithm Usage Checker",
                    "how_to_fix": "Use deterministic algorithms for reproducibility.",
                    "benefits": "Ensures consistent results across different runs.",
                    "strategies": "Set torch.use_deterministic_algorithms(True)."
                },
                {
                    "name": "Randomness Control Checker (PyTorch-Dataloader)",
                    "how_to_fix": "Set worker_init_fn and generator in DataLoader.",
                    "benefits": "Ensures reproducible data loading and augmentation.",
                    "strategies": "Configure worker_init_fn and generator in DataLoader initialization."
                },
                {
                    "name": "Mask Missing Checker",
                    "how_to_fix": "Ensure valid arguments for log functions.",
                    "benefits": "Prevents numerical errors and improves model stability.",
                    "strategies": "Validate arguments before using log functions."
                },
                {
                    "name": "Net Forward Checker",
                    "how_to_fix": "Use self.net() instead of self.net.forward().",
                    "benefits": "Improves code readability and maintains consistency with PyTorch practices.",
                    "strategies": "Replace self.net.forward() with self.net()."
                },
                {
                    "name": "Gradient Clear Checker",
                    "how_to_fix": "Use optimizer.zero_grad() before loss.backward() and optimizer.step().",
                    "benefits": "Ensures correct gradient computation and prevents accumulation.",
                    "strategies": "Include optimizer.zero_grad() in the training loop."
                },
                {
                    "name": "Batch Normalisation Checker",
                    "how_to_fix": "Use batch normalisation layers to improve training stability.",
                    "benefits": "Enhances model performance and convergence speed.",
                    "strategies": "Incorporate torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, etc., into the model."
                },
                {
                    "name": "Dropout Usage Checker",
                    "how_to_fix": "Use dropout layers to mitigate overfitting.",
                    "benefits": "Prevents overfitting and enhances generalisation of the model.",
                    "strategies": "Add torch.nn.Dropout layers to the neural network."
                },
                {
                    "name": "Data Augmentation Checker",
                    "how_to_fix": "Implement data augmentation to increase dataset diversity.",
                    "benefits": "Enhances model performance and reduces overfitting.",
                    "strategies": "Use torchvision.transforms for applying data augmentation techniques."
                },
                {
                    "name": "Learning Rate Scheduler Checker",
                    "how_to_fix": "Use learning rate schedulers for better training optimisation.",
                    "benefits": "Enhances training efficiency and model accuracy.",
                    "strategies": "Implement torch.optim.lr_scheduler for dynamic learning rate adjustments."
                },
                {
                    "name": "Logging Checker",
                    "how_to_fix": "Use logging for tracking experiments and results.",
                    "benefits": "Helps in tracking training progress and diagnosing issues.",
                    "strategies": "Use logging frameworks such as tensorboardX or logging module in PyTorch."
                },
                {
                    "name": "Model Evaluation Checker",
                    "how_to_fix": "Continuously evaluate model on validation/test datasets.",
                    "benefits": "Prevents overfitting and ensures robust performance.",
                    "strategies": "Implement periodic evaluation logic within the training loop."
                },
                {
                    "name": "Unit Testing Checker",
                    "how_to_fix": "Implement unit tests for PyTorch components and models.",
                    "benefits": "Improves code quality and reduces chances of errors.",
                    "strategies": "Write unit tests using unittest or pytest for PyTorch code."
                },
                {
                    "name": "Exception Handling Checker",
                    "how_to_fix": "Use exception handling to manage potential errors during model operations.",
                    "benefits": "Ensures graceful handling of errors and robustness of the code.",
                    "strategies": "Implement try-except blocks to catch and manage exceptions in PyTorch code."
                }
            ]
        }

    def generate_report(self) -> str:
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
            
            # Only show line number if it's not 0
            if smell['line_number'] != 0:
                report += f"Line: {smell['line_number']}\n"
            
            # Only include code snippet if it's 3 lines or fewer
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
