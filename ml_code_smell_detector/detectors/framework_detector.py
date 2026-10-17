import sys
from pathlib import Path
from typing import Any, Dict, List

import astroid
from astroid import nodes
from astroid.const import Context

from ml_code_smell_detector.utils import (
    callee_name,
    get_import_map,
    in_loop,
    int_value,
    keyword,
    nodes_of,
    parse_file,
    resolve_call_name,
    run_checks,
    source,
)


class FrameworkSpecificSmellDetector:
    """A detector for identifying code smells specific to ML frameworks like
    Pandas, NumPy, Scikit-learn, PyTorch and TensorFlow.

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
            module = parse_file(file_path)

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
                print(
                    f"Skipping framework-specific smell detection for {file_path}: No relevant frameworks imported",
                    file=sys.stderr)
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
            for import_node in nodes_of(node, (nodes.Import, nodes.ImportFrom)):
                if isinstance(import_node, nodes.Import):
                    for name, _ in import_node.names:
                        root = name.split('.')[0]
                        if root in framework_imports and framework_imports[root] not in frameworks:
                            frameworks.append(framework_imports[root])
                elif isinstance(import_node, nodes.ImportFrom):
                    root = import_node.modname.split('.')[0]
                    if root in framework_imports and framework_imports[root] not in frameworks:
                        frameworks.append(framework_imports[root])
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
                "code_snippet": source(node),
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
        run_checks((
            self.detect_iterrows_usage,
            self.detect_chain_indexing,
            self.detect_merge_parameters,
            self.detect_inplace_operations,
            self.detect_values_usage,
            self.detect_dtype_specification,
            self.detect_column_selection,
            self.detect_dataframe_modification,
        ), node, file_path)

    def detect_iterrows_usage(self, node: nodes.Module, file_path: str):
        # Only flag iterrows if better alternatives could be used
        vectorizable_operations = ['sum', 'mean', 'max', 'min', 'apply', 'map']
        for call in nodes_of(node, nodes.Call):
            if 'iterrows' in source(call.func):
                # Check if the iterrows is used within a loop
                parent = call.parent
                while parent and not isinstance(parent, nodes.For):
                    parent = parent.parent
                if parent and any(op in source(parent) for op in vectorizable_operations):
                    self.add_smell('Pandas', 'Unnecessary Iteration', call, file_path)

    def detect_chain_indexing(self, node: nodes.Module, file_path: str):
        for subscript in nodes_of(node, nodes.Subscript):
            if isinstance(subscript.value, nodes.Subscript):
                # Only flag if it's not within a with pd.option_context block
                parent = subscript.parent
                while parent and not isinstance(parent, nodes.With):
                    parent = parent.parent
                if not parent or 'option_context' not in source(parent):
                    # Only chained assignment/deletion (df['a'][0] = v, df[mask]['a'] = v) is the smell; chained
                    # reads are fine. A column label on either side tells it apart from nested lists (grid[i][j] = v).
                    if subscript.ctx != Context.Load and any(
                            isinstance(s, nodes.Const) and isinstance(s.value, str)
                            for s in (subscript.slice, subscript.value.slice)):
                        self.add_smell('Pandas', 'Chain Indexing', subscript, file_path)

    MERGE_KEYWORDS = {'how', 'on', 'left_on', 'right_on', 'left_index', 'right_index', 'validate'}

    def detect_merge_parameters(self, node: nodes.Module, file_path: str):
        # pd.merge(a, b) and the (more common) a.merge(b); merges resolved to another module (heapq.merge) are not pandas
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            if callee_name(call) != 'merge' or not isinstance(call.func, nodes.Attribute):
                continue
            qualname = resolve_call_name(call, import_map)
            if qualname and not qualname.startswith('pandas.'):
                continue
            if not any(kw.arg in self.MERGE_KEYWORDS or kw.arg is None for kw in call.keywords):
                self.add_smell('Pandas', 'Merge Parameter Checker', call, file_path)

    def detect_inplace_operations(self, node: nodes.Module, file_path: str):
        # Every inplace=True is reported; `df2 = df.dropna(inplace=True)` is the worst case (df2 is None)
        for call in nodes_of(node, nodes.Call):
            value = keyword(call, 'inplace')
            if isinstance(value, nodes.Const) and value.value is True:
                self.add_smell('Pandas', 'InPlace Checker', call, file_path)

    def detect_values_usage(self, node: nodes.Module, file_path: str):
        # The `.values` attribute (not the dict.values() method), reported once, when used in a numpy context
        numpy_contexts = ['np.', 'array', 'asarray', 'reshape', 'transpose']
        for attr in nodes_of(node, nodes.Attribute):
            if attr.attrname != 'values' or (isinstance(attr.parent, nodes.Call) and attr.parent.func is attr):
                continue
            if any(isinstance(anc, nodes.Call) and any(ctx in source(anc.func) for ctx in numpy_contexts)
                   for anc in attr.node_ancestors()):
                self.add_smell('Pandas', 'DataFrame Conversion Checker', attr, file_path)

    DTYPE_SENSITIVE_OPS = {'groupby', 'merge', 'sort_values', 'pivot_table', 'astype'}

    def detect_dtype_specification(self, node: nodes.Module, file_path: str):
        # pd.read_csv without dtype=, in a module that then does dtype-sensitive work (grouping, merging, sorting)
        import_map = get_import_map(node)
        if not any(callee_name(c) in self.DTYPE_SENSITIVE_OPS for c in nodes_of(node, nodes.Call)):
            return
        for call in nodes_of(node, nodes.Call):
            if (resolve_call_name(call, import_map) == 'pandas.read_csv' and
                    not any(kw.arg in ('dtype', 'parse_dates', 'converters') or kw.arg is None for kw in call.keywords)):
                self.add_smell('Pandas', 'Datatype Checker', call, file_path)

    PANDAS_READERS = {'read_csv', 'read_table', 'read_excel', 'read_parquet', 'read_feather', 'read_sql', 'read_sql_table'}

    def detect_column_selection(self, node: nodes.Module, file_path: str):
        # A file read without usecols/columns whose columns are never narrowed down afterwards (df[['a', 'b']])
        import_map = get_import_map(node)
        selects_columns = any(
            isinstance(s.slice, nodes.List) or isinstance(s.slice, nodes.Tuple) and any(isinstance(e, nodes.List) for e in s.slice.elts)
            for s in nodes_of(node, nodes.Subscript)
        ) or any(callee_name(c) in ('filter', 'reindex') for c in nodes_of(node, nodes.Call))
        if selects_columns:
            return
        for call in nodes_of(node, nodes.Call):
            qualname = resolve_call_name(call, import_map) or ''
            if (qualname.startswith('pandas.') and qualname.rsplit('.', 1)[-1] in self.PANDAS_READERS and
                    not any(kw.arg in ('usecols', 'columns') or kw.arg is None for kw in call.keywords)):
                self.add_smell('Pandas', 'Column Selection Checker', call, file_path)

    ROW_ITERATORS = {'iterrows', 'itertuples'}

    def detect_dataframe_modification(self, node: nodes.Module, file_path: str):
        # Writing to the frame being iterated (df.loc[i, 'a'] = v inside `for i, row in df.iterrows()`), or to
        # the row copy (row['a'] = v, which silently has no effect on df)
        for loop in nodes_of(node, nodes.For):
            it = loop.iter
            if not (isinstance(it, nodes.Call) and isinstance(it.func, nodes.Attribute) and it.func.attrname in self.ROW_ITERATORS):
                continue
            frame = source(it.func.expr)
            rows = {n.name for n in nodes_of(loop.target, nodes.AssignName)} if loop.target else set()
            for stmt in nodes_of(loop, (nodes.Assign, nodes.AugAssign)):
                targets = stmt.targets if isinstance(stmt, nodes.Assign) else [stmt.target]
                for target in targets:
                    if not isinstance(target, nodes.Subscript):
                        continue
                    base = target.value
                    if isinstance(base, nodes.Attribute) and base.attrname in ('loc', 'iloc', 'at', 'iat'):
                        base = base.expr
                    if source(base) == frame or (isinstance(base, nodes.Name) and base.name in rows):
                        self.add_smell('Pandas', 'DataFrame Iteration Modification', stmt, file_path)

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
        run_checks((
            self.detect_nan_equality,
            self.detect_random_seed,
            self.detect_array_creation,
            self.detect_inefficient_operations,
            self.detect_dtype_consistency,
            self.detect_broadcasting_issues,
            self.detect_copy_view_issues,
            self.detect_axis_specification,
        ), node, file_path)

    NAN_NAMES = {'numpy.nan', 'numpy.NaN', 'numpy.NAN', 'math.nan', 'pandas.NA'}

    def detect_nan_equality(self, node: nodes.Module, file_path: str):
        # x == np.nan is always False (and != always True); either side, and also float('nan')
        import_map = get_import_map(node)

        def is_nan(operand):
            if isinstance(operand, nodes.Call):
                return (isinstance(operand.func, nodes.Name) and operand.func.name == 'float' and len(operand.args) == 1 and
                        isinstance(operand.args[0], nodes.Const) and str(operand.args[0].value).lower() == 'nan')
            parts = []
            while isinstance(operand, nodes.Attribute):
                parts.append(operand.attrname)
                operand = operand.expr
            if not isinstance(operand, nodes.Name) or operand.name not in import_map:
                return False
            return '.'.join([import_map[operand.name]] + parts[::-1]) in self.NAN_NAMES

        for compare in nodes_of(node, nodes.Compare):
            operands = [compare.left] + [right for _, right in compare.ops]
            if any(op in ('==', '!=') for op, _ in compare.ops) and any(is_nan(o) for o in operands):
                self.add_smell('NumPy', 'NaN Equality Checker', compare, file_path)

    # numpy.random functions that do not draw from the global generator
    NUMPY_RANDOM_NON_DRAWS = {'seed', 'default_rng', 'RandomState', 'Generator', 'SeedSequence', 'get_state', 'set_state',
                              'PCG64', 'PCG64DXSM', 'MT19937', 'Philox', 'SFC64', 'BitGenerator'}

    def detect_random_seed(self, node: nodes.Module, file_path: str):
        # Draws from the global generator without np.random.seed(), reported at the first draw; and generators
        # created without a seed (np.random.default_rng()), reported where they are created
        import_map = get_import_map(node)
        draws, seeded = [], False
        for call in nodes_of(node, nodes.Call):
            qualname = resolve_call_name(call, import_map) or ''
            if not qualname.startswith('numpy.random.'):
                continue
            name = qualname.rsplit('.', 1)[-1]
            if name == 'seed':
                seeded = True
            elif name in ('default_rng', 'RandomState'):
                if not call.args and not call.keywords:
                    self.add_smell('NumPy', 'Randomness Control Checker', call, file_path)
            elif name not in self.NUMPY_RANDOM_NON_DRAWS:
                draws.append(call)
        if draws and not seeded:
            self.add_smell('NumPy', 'Randomness Control Checker', draws[0], file_path)

    ARRAY_CONSTRUCTORS = {'numpy.array', 'numpy.zeros', 'numpy.ones', 'numpy.empty'}

    def detect_array_creation(self, node: nodes.Module, file_path: str):
        """Detect array creation without dtype (exact constructors; ``*_like`` inherit their argument's dtype)"""
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            if (resolve_call_name(call, import_map) in self.ARRAY_CONSTRUCTORS and
                    not any(kw.arg == 'dtype' for kw in call.keywords)):
                self.add_smell('NumPy', 'Array Creation Efficiency', call, file_path)

    def detect_inefficient_operations(self, node: nodes.Module, file_path: str):
        """Detect inefficient numerical operations"""
        # Report the innermost loop around each reduction once (outer loops would duplicate the same call)
        import_map = get_import_map(node)
        reported = []
        for call in nodes_of(node, nodes.Call):
            if resolve_call_name(call, import_map) in ('numpy.sum', 'numpy.mean', 'numpy.max', 'numpy.min'):
                loop = next((a for a in call.node_ancestors() if isinstance(a, nodes.For)), None)
                if loop is not None and not any(loop is r for r in reported):
                    reported.append(loop)
                    self.add_smell('NumPy', 'Inefficient Operations', loop, file_path)

        # Growing an array inside a loop copies it every iteration (quadratic); collect in a list, join once
        for call in nodes_of(node, nodes.Call):
            if resolve_call_name(call, import_map) in self.ARRAY_GROWERS and in_loop(call):
                self.add_smell('NumPy', 'Inefficient Operations', call, file_path)

    ARRAY_GROWERS = {'numpy.concatenate', 'numpy.append', 'numpy.vstack', 'numpy.hstack', 'numpy.stack', 'numpy.row_stack'}

    INT_SCALARS = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'intc', 'intp'}
    FLOAT_SCALARS = {'float16', 'float32', 'float64', 'half', 'single', 'double', 'longdouble'}

    def detect_dtype_consistency(self, node: nodes.Module, file_path: str):
        """Detect potential dtype inconsistency issues"""
        import_map = get_import_map(node)

        def scalar_kind(operand):
            qualname = (resolve_call_name(operand, import_map) or '') if isinstance(operand, nodes.Call) else ''
            if not qualname.startswith('numpy.'):
                return None
            name = qualname.rsplit('.', 1)[-1]
            return 'int' if name in self.INT_SCALARS else 'float' if name in self.FLOAT_SCALARS else None

        for binop in nodes_of(node, nodes.BinOp):
            if {scalar_kind(binop.left), scalar_kind(binop.right)} == {'int', 'float'}:
                self.add_smell('NumPy', 'Dtype Consistency', binop, file_path)

    def detect_broadcasting_issues(self, node: nodes.Module, file_path: str):
        """Detect potential broadcasting issues"""
        for binop in nodes_of(node, nodes.BinOp):
            if isinstance(binop.left, nodes.Call) and isinstance(binop.right, nodes.Call):
                # Check for operations between arrays that might have broadcasting issues
                if ('reshape' in source(binop.left) or 'reshape' in source(binop.right) or
                        'transpose' in source(binop.left) or 'transpose' in source(binop.right)):
                    self.add_smell('NumPy', 'Broadcasting Risk', binop, file_path)

    def detect_copy_view_issues(self, node: nodes.Module, file_path: str):
        """Detect `view = arr[a:b]` (basic slicing returns a view) followed by an in-place write to `view`,
        which silently modifies `arr` too. Only arrays built by a numpy call in the same scope are tracked."""
        import_map = get_import_map(node)

        def is_basic_slice(index):
            elts = index.elts if isinstance(index, nodes.Tuple) else [index]
            return any(isinstance(e, nodes.Slice) for e in elts) and not any(isinstance(e, (nodes.List, nodes.Compare)) for e in elts)

        for scope in [node] + list(nodes_of(node, nodes.FunctionDef)):
            def own(n):
                return n.scope() is scope
            arrays = {a.targets[0].name for a in nodes_of(scope, nodes.Assign)
                      if own(a) and isinstance(a.targets[0], nodes.AssignName) and isinstance(a.value, nodes.Call) and
                      (resolve_call_name(a.value, import_map) or '').startswith('numpy.')}
            for assign in nodes_of(scope, nodes.Assign):
                value, target = assign.value, assign.targets[0]
                # `a = a[1:]` rebinds the name: no second alias of the buffer is left to be confused about
                if not (own(assign) and isinstance(target, nodes.AssignName) and isinstance(value, nodes.Subscript) and
                        isinstance(value.value, nodes.Name) and value.value.name in arrays and is_basic_slice(value.slice) and
                        value.value.name != target.name):
                    continue
                writes = [w for w in nodes_of(scope, (nodes.Assign, nodes.AugAssign)) if own(w) and w.lineno > assign.lineno]
                for w in writes:
                    targets = w.targets if isinstance(w, nodes.Assign) else [w.target]
                    if any(isinstance(t, nodes.Subscript) and isinstance(t.value, nodes.Name) and t.value.name == target.name or
                           isinstance(w, nodes.AugAssign) and isinstance(t, nodes.AssignName) and t.name == target.name
                           for t in targets):
                        self.add_smell('NumPy', 'Copy-View Confusion', assign, file_path)
                        break

    AXIS_REDUCTIONS = {'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'any', 'all', 'std', 'var', 'median', 'prod',
                       'nansum', 'nanmean', 'nanmax', 'nanmin', 'nanstd', 'nanvar', 'nanmedian', 'amax', 'amin'}

    def detect_axis_specification(self, node: nodes.Module, file_path: str):
        """Detect numpy reductions called without axis (they silently reduce over the flattened array)"""
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            qualname = resolve_call_name(call, import_map) or ''
            if (qualname.startswith('numpy.') and qualname.count('.') == 1 and qualname[6:] in self.AXIS_REDUCTIONS and
                    not any(kw.arg in ('axis', None) for kw in call.keywords) and
                    len(call.args) < 2):  # axis can also be specified as positional argument
                self.add_smell('NumPy', 'Missing Axis Specification', call, file_path)

    # Scikit-learn Detection Methods
    def detect_scikitlearn_smells(self, node: nodes.Module, file_path: str):
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
        run_checks((
            self.detect_scaling_usage,
            self.detect_pipeline_usage,
            self.detect_cross_validation,
            self.detect_random_state,
            self.detect_verbose_mode,
            self.detect_threshold_metrics,
            self.detect_unit_tests,
            self.detect_data_leakage,
            self.detect_exception_handling,
        ), node, file_path)

    # Distance-, margin-, gradient- or variance-based estimators whose results depend on feature scale
    SCALE_SENSITIVE = {'SVC', 'SVR', 'LinearSVC', 'LinearSVR', 'NuSVC', 'NuSVR', 'KNeighborsClassifier', 'KNeighborsRegressor',
                       'RadiusNeighborsClassifier', 'RadiusNeighborsRegressor', 'KMeans', 'MiniBatchKMeans', 'PCA', 'KernelPCA',
                       'LogisticRegression', 'SGDClassifier', 'SGDRegressor', 'MLPClassifier', 'MLPRegressor',
                       'Ridge', 'Lasso', 'ElasticNet', 'DBSCAN', 'SpectralClustering', 'GaussianProcessClassifier'}
    SCALERS = {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'Normalizer', 'QuantileTransformer',
               'PowerTransformer', 'scale', 'minmax_scale', 'robust_scale', 'maxabs_scale', 'normalize'}

    def detect_scaling_usage(self, node: nodes.Module, file_path: str):
        # Reported at the first scale-sensitive estimator when nothing in the module scales features
        if any(callee_name(c) in self.SCALERS for c in nodes_of(node, nodes.Call)):
            return
        first = next((c for c in nodes_of(node, nodes.Call) if callee_name(c) in self.SCALE_SENSITIVE), None)
        if first is not None:
            self.add_smell('ScikitLearn', 'Scaler Missing Checker', first, file_path)

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
            step in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for step in preprocessing_steps
        )
        has_model_steps = any(
            step in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for step in model_steps
        )

        # Only suggest Pipeline if multiple steps are present (make_pipeline(...) is a Pipeline too)
        if has_preprocessing and has_model_steps and not any(
            callee_name(call) in ('Pipeline', 'make_pipeline')
            for call in nodes_of(node, nodes.Call)
        ):
            self.add_smell('ScikitLearn', 'Pipeline Checker', node, file_path)

    def detect_cross_validation(self, node: nodes.Module, file_path: str):
        # Check if model training is performed
        training_indicators = ['fit', 'train']
        has_training = any(
            indicator in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for indicator in training_indicators
        )

        # Check for any cross-validation technique
        # Substrings: 'KFold' covers Stratified/Group/Repeated variants, 'SearchCV' the Halving* searches,
        # 'CV' the built-in CV estimators (LogisticRegressionCV, RidgeCV, RFECV, ...)
        cv_methods = [
            'cross_val_score', 'cross_val_predict', 'cross_validate', 'KFold', 'ShuffleSplit', 'TimeSeriesSplit',
            'LeaveOneOut', 'LeavePOut', 'LeaveOneGroupOut', 'PredefinedSplit', 'SearchCV', 'CV',
        ]
        has_cv = any(
            method in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for method in cv_methods
        )

        # Only suggest cross-validation for model training scenarios
        if has_training and not has_cv:
            self.add_smell('ScikitLearn', 'Cross Validation Checker', node, file_path)

    # scikit-learn APIs whose result depends on random_state (matched on the resolved callee, not a substring)
    RANDOM_STATE_APIS = {
        'train_test_split', 'shuffle', 'resample', 'KFold', 'StratifiedKFold', 'RepeatedKFold', 'RepeatedStratifiedKFold',
        'ShuffleSplit', 'StratifiedShuffleSplit', 'GroupShuffleSplit', 'RandomizedSearchCV',
        'RandomForestClassifier', 'RandomForestRegressor', 'ExtraTreesClassifier', 'ExtraTreesRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor', 'HistGradientBoostingClassifier',
        'HistGradientBoostingRegressor', 'AdaBoostClassifier', 'AdaBoostRegressor', 'BaggingClassifier', 'BaggingRegressor',
        'IsolationForest', 'DecisionTreeClassifier', 'DecisionTreeRegressor', 'MLPClassifier', 'MLPRegressor',
        'SGDClassifier', 'SGDRegressor', 'KMeans', 'MiniBatchKMeans', 'PCA', 'KernelPCA', 'SparsePCA', 'MiniBatchSparsePCA',
        'TruncatedSVD', 'NMF', 'TSNE', 'GaussianMixture', 'make_classification', 'make_regression', 'make_blobs',
    }
    # These splitters only use random_state when shuffle=True (sklearn raises if it is set otherwise)
    SHUFFLE_GATED_SPLITTERS = {'KFold', 'StratifiedKFold'}

    def detect_random_state(self, node: nodes.Module, file_path: str):
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            qualname = resolve_call_name(call, import_map) or ''
            api = qualname.rsplit('.', 1)[-1]
            if not qualname.startswith('sklearn.') or api not in self.RANDOM_STATE_APIS:
                continue
            if api in self.SHUFFLE_GATED_SPLITTERS and not any(
                    kw.arg == 'shuffle' and not (isinstance(kw.value, nodes.Const) and kw.value.value is False)
                    for kw in call.keywords):
                continue
            if not any(kw.arg in ('random_state', None) for kw in call.keywords):  # None: **kwargs may carry it
                self.add_smell('ScikitLearn', 'Randomness Control Checker', call, file_path)

    # Long-running scikit-learn constructors that take a verbose parameter (fit() itself never does)
    VERBOSE_APIS = {
        'GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV', 'HalvingRandomSearchCV',
        'RandomForestClassifier', 'RandomForestRegressor', 'GradientBoostingClassifier', 'GradientBoostingRegressor',
    }

    def detect_verbose_mode(self, node: nodes.Module, file_path: str):
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            qualname = resolve_call_name(call, import_map) or ''
            if (qualname.startswith('sklearn.') and qualname.rsplit('.', 1)[-1] in self.VERBOSE_APIS and
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
            indicator in source(node)
            for indicator in classification_indicators
        )

        # Check for threshold-independent metrics
        threshold_independent_metrics = [
            'roc_auc_score', 'average_precision_score',
            'precision_recall_curve', 'roc_curve'
        ]
        has_threshold_metrics = any(
            metric in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for metric in threshold_independent_metrics
        )

        # Only suggest threshold-independent metrics for classification tasks
        if is_classification and not has_threshold_metrics:
            self.add_smell('ScikitLearn', 'Dependent Threshold Checker', node, file_path)

    @staticmethod
    def _has_test_file(path: Path) -> bool:
        """True if test_<stem>.py / <stem>_test.py exists next to the file or in a tests/ or test/ dir of an ancestor."""
        names = (f"test_{path.stem}.py", f"{path.stem}_test.py")
        dirs = [path.parent] + [d / sub for d in path.resolve().parents for sub in ('tests', 'test')]
        return any((d / n).is_file() for d in dirs for n in names)

    def detect_unit_tests(self, node: nodes.Module, file_path: str, framework: str = 'ScikitLearn'):
        path = Path(file_path)
        # A test module is recognised by its name/location or by the tests it defines, not by the substring 'test'
        is_test_file = (path.name.startswith('test_') or path.stem.endswith('_test') or
                        any(part in ('test', 'tests') for part in path.parent.parts) or
                        any(f.name.startswith('test_') for f in nodes_of(node, nodes.FunctionDef)) or
                        any('TestCase' in source(base) for c in nodes_of(node, nodes.ClassDef) for base in c.bases))

        # Check for model training or evaluation code
        ml_operations = [
            'fit', 'predict', 'transform', 'score',
            'cross_val_score', 'GridSearchCV'
        ]
        has_ml_operations = any(
            op in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for op in ml_operations
        )

        # Tests live in their own files: look for one instead of grepping this module for 'pytest'
        if not is_test_file and has_ml_operations and not self._has_test_file(path):
            self.add_smell(framework, 'Unit Testing Checker', node, file_path)

    def detect_data_leakage(self, node: nodes.Module, file_path: str):
        # Check for data preprocessing operations
        preprocessing_ops = [
            'fit_transform', 'transform', 'StandardScaler',
            'MinMaxScaler', 'PCA', 'feature_selection'
        ]
        has_preprocessing = any(
            op in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for op in preprocessing_ops
        )

        # Check for model training
        has_model_training = any(
            'fit' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        # Check for proper train-test splitting
        splitting_methods = [
            'train_test_split', 'KFold', 'StratifiedKFold',
            'GroupKFold', 'TimeSeriesSplit'
        ]
        has_proper_split = any(
            method in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for method in splitting_methods
        )

        # Only report if preprocessing and training without proper splitting
        if (has_preprocessing and has_model_training and not has_proper_split):
            self.add_smell('ScikitLearn', 'Data Leakage Checker', node, file_path)

    # Model persistence I/O: the failures here (missing/corrupt/incompatible file) are worth handling.
    # fit/predict/transform are deliberately excluded: wrapping them in try/except mostly hides real errors.
    RISKY_IO_CALLS = {'joblib.load', 'joblib.dump', 'pickle.load', 'pickle.dump'}

    def detect_exception_handling(self, node: nodes.Module, file_path: str, framework: str = 'ScikitLearn', is_risky=None):
        import_map = get_import_map(node)
        is_risky = is_risky or (lambda qualname: qualname in self.RISKY_IO_CALLS)
        for call in nodes_of(node, nodes.Call):
            if not is_risky(resolve_call_name(call, import_map) or ''):
                continue
            if not any(isinstance(parent, nodes.Try) for parent in call.node_ancestors()):
                self.add_smell(framework, 'Exception Handling Checker', call, file_path)

    @staticmethod
    def _is_tf_artifact_io(qualname: str) -> bool:
        """tf.keras.models.load_model / keras.models.load_model / tf.saved_model.load|save, however imported."""
        return (qualname.startswith(('tensorflow.', 'keras.')) and
                qualname.endswith(('models.load_model', 'saved_model.load', 'saved_model.save')))

    def detect_tf_unit_tests(self, node: nodes.Module, file_path: str):
        self.detect_unit_tests(node, file_path, framework='TensorFlow')

    def detect_tf_exception_handling(self, node: nodes.Module, file_path: str):
        self.detect_exception_handling(node, file_path, framework='TensorFlow', is_risky=self._is_tf_artifact_io)

    def check_imports(self, node: nodes.Module, file_path: str):
        # The np/pd alias convention is about importing the module; `from numpy import ndarray` is fine
        for import_node in nodes_of(node, nodes.Import):
            for name, alias in import_node.names:
                if name == 'numpy' and alias != 'np':
                    self.add_smell('General', 'Import Checker', import_node, file_path)
                elif name == 'pandas' and alias != 'pd':
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
        run_checks((
            self.detect_pytorch_random_seed,
            self.detect_pytorch_deterministic,
            self.detect_pytorch_dataloader_random,
            self.detect_pytorch_mask,
            self.detect_pytorch_forward,
            self.detect_pytorch_grad_zero,
            self.detect_pytorch_batch_norm,
            self.detect_pytorch_dropout,
            self.detect_pytorch_augmentation,
            self.detect_pytorch_lr_scheduler,
            self.detect_pytorch_logging,
            self.detect_pytorch_eval_mode,
        ), node, file_path)

    TORCH_SEEDERS = ('manual_seed', 'manual_seed_all', 'seed_everything', 'set_seed')

    def detect_pytorch_random_seed(self, node: nodes.Module, file_path: str):
        # Random tensor ops, dropout and random_split need torch.manual_seed (or a seed_everything/set_seed helper)
        random_operations = [
            'torch.rand', 'torch.randn', 'torch.randint', 'torch.randperm', 'torch.bernoulli', 'torch.normal',
            'torch.multinomial', 'dropout', 'Dropout', 'random_split',
        ]

        has_random_ops = any(
            op in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for op in random_operations
        )

        has_manual_seed = any(callee_name(call) in self.TORCH_SEEDERS for call in nodes_of(node, nodes.Call))

        # Only report if random operations are used without seeding
        if has_random_ops and not has_manual_seed:
            self.add_smell('PyTorch', 'Randomness Control Checker', node, file_path)

    def detect_pytorch_deterministic(self, node: nodes.Module, file_path: str):
        # Check for operations that benefit from deterministic algorithms
        # torch.cuda alone (e.g. torch.cuda.is_available()) is device selection, not a nondeterministic op
        deterministic_sensitive_ops = [
            'torch.nn.Conv', 'nn.Conv', 'torch.nn.LSTM', 'nn.LSTM', 'torch.nn.GRU', 'nn.GRU',
            'torch.backends.cudnn', 'DataLoader'
        ]

        has_sensitive_ops = any(
            op in source(node)
            for op in deterministic_sensitive_ops
        )

        # cudnn.deterministic is set by assignment (also after `from torch.backends import cudnn`)
        has_deterministic_setting = any(
            callee_name(call) == 'use_deterministic_algorithms'
            for call in nodes_of(node, nodes.Call)
        ) or any(
            source(target).endswith('cudnn.deterministic')
            for assign in nodes_of(node, nodes.Assign)
            for target in assign.targets
        )

        # Only report if sensitive operations are used without deterministic setting
        if has_sensitive_ops and not has_deterministic_setting:
            self.add_smell('PyTorch', 'Deterministic Algorithm Usage Checker', node, file_path)

    def detect_pytorch_dataloader_random(self, node: nodes.Module, file_path: str):
        # A shuffling torch DataLoader without generator=/worker_init_fn= (however DataLoader was imported)
        import_map = get_import_map(node)
        for call in nodes_of(node, nodes.Call):
            if callee_name(call) != 'DataLoader' or not (resolve_call_name(call, import_map) or '').startswith('torch.'):
                continue
            shuffle = keyword(call, 'shuffle')
            has_shuffle = not (shuffle is None or isinstance(shuffle, nodes.Const) and not shuffle.value)
            has_random_control = any(kw.arg in ('worker_init_fn', 'generator', 'sampler', 'batch_sampler', None)
                                     for kw in call.keywords)
            if has_shuffle and not has_random_control:
                self.add_smell('PyTorch', 'Randomness Control Checker (PyTorch-Dataloader)', call, file_path)

    TORCH_LOGS = {'torch.log', 'torch.log2', 'torch.log10'}

    def detect_pytorch_mask(self, node: nodes.Module, file_path: str):
        # torch.log(x) of an input that can be zero/negative, with no clamp/mask/where in the module.
        # Exact callee: log_softmax, logsumexp, logical_and and log1p are safe or unrelated.
        import_map = get_import_map(node)
        has_mask = any(op in source(node) for op in ('clamp', 'mask', 'where'))
        if has_mask:
            return
        risky_ops = ['zeros', 'randn', 'rand', 'sub', 'subtract']
        for call in nodes_of(node, nodes.Call):
            if resolve_call_name(call, import_map) in self.TORCH_LOGS and len(call.args) == 1:
                if any(op in source(call.args[0]) for op in risky_ops):
                    self.add_smell('PyTorch', 'Mask Missing Checker', call, file_path)

    def detect_pytorch_forward(self, node: nodes.Module, file_path: str):
        # model.forward(x) / self.encoder.forward(x) bypass nn.Module.__call__ (hooks). Not reported: forward on a
        # call result (super().forward(x) in an override, rref.rpc_async().forward(x) RPC proxies) and a module
        # calling its own self.forward(x).
        for call in nodes_of(node, nodes.Call):
            func = call.func
            if not (isinstance(func, nodes.Attribute) and func.attrname == 'forward'):
                continue
            receiver = func.expr
            if isinstance(receiver, nodes.Call) or isinstance(receiver, nodes.Name) and receiver.name == 'self':
                continue
            self.add_smell('PyTorch', 'Net Forward Checker', call, file_path)

    TORCH_OPTIMIZERS = {'SGD', 'Adam', 'AdamW', 'Adamax', 'Adagrad', 'Adadelta', 'RMSprop', 'Rprop', 'NAdam', 'RAdam',
                        'LBFGS', 'SparseAdam', 'ASGD'}

    def _has_torch_optimizer(self, node: nodes.Module) -> bool:
        import_map = get_import_map(node)
        return any('optim.' in source(call.func) or
                   (resolve_call_name(call, import_map) or '').startswith('torch.optim') or
                   callee_name(call) in self.TORCH_OPTIMIZERS
                   for call in nodes_of(node, nodes.Call))

    def detect_pytorch_grad_zero(self, node: nodes.Module, file_path: str):
        # loss.backward() with an optimizer but no zero_grad() anywhere: gradients accumulate across steps
        backward = next((c for c in nodes_of(node, nodes.Call) if callee_name(c) == 'backward'), None)
        has_grad_zero = any(callee_name(c) == 'zero_grad' for c in nodes_of(node, nodes.Call))
        if backward is not None and not has_grad_zero and self._has_torch_optimizer(node):
            self.add_smell('PyTorch', 'Gradient Clear Checker', backward, file_path)

    def detect_pytorch_batch_norm(self, node: nodes.Module, file_path: str):
        # Check if it's a CNN or deep network that could benefit from BatchNorm
        conv_layers = [
            'Conv1d', 'Conv2d', 'Conv3d',
            'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'
        ]

        has_conv_layers = any(
            layer in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for layer in conv_layers
        )

        has_deep_structure = len([
            call for call in nodes_of(node, nodes.Call)
            if any(layer in source(call.func) for layer in ['Linear', 'Conv'])
        ]) > 2

        # Any normalisation layer counts: BatchNorm, LayerNorm (transformers), GroupNorm, InstanceNorm, RMSNorm
        has_batch_norm = any(
            'Norm' in callee_name(call)
            for call in nodes_of(node, nodes.Call)
        )

        # Only suggest BatchNorm for appropriate architectures
        if (has_conv_layers or has_deep_structure) and not has_batch_norm:
            self.add_smell('PyTorch', 'Batch Normalisation Checker', node, file_path)

    def detect_pytorch_dropout(self, node: nodes.Module, file_path: str):
        # Check if the model is complex enough to benefit from dropout
        has_multiple_layers = len([
            call for call in nodes_of(node, nodes.Call)
            if any(layer in source(call.func)
                   for layer in ['Linear', 'Conv', 'LSTM', 'GRU'])
        ]) > 2

        has_training_code = any(
            callee_name(call) in ('train', 'backward')
            for call in nodes_of(node, nodes.Call)
        )

        has_dropout = any(
            'dropout' in source(call.func).lower()
            for call in nodes_of(node, nodes.Call)
        )

        # Only suggest dropout for complex models during training
        if has_multiple_layers and has_training_code and not has_dropout:
            self.add_smell('PyTorch', 'Dropout Usage Checker', node, file_path)

    def detect_pytorch_augmentation(self, node: nodes.Module, file_path: str):
        # A vision task: image datasets/layers/libraries. DataLoader or Dataset alone says nothing about images.
        vision_indicators = [
            'ImageFolder', 'torchvision', 'Conv2d', 'Conv3d', 'ResNet', 'resnet', 'VGG', 'vgg', 'PIL', 'Image.open',
            'CIFAR', 'MNIST', 'ImageNet',
        ]

        is_vision_task = any(
            indicator in source(node)
            for indicator in vision_indicators
        )

        has_training = any(
            callee_name(call) in ('train', 'fit', 'backward')
            for call in nodes_of(node, nodes.Call)
        )

        has_augmentation = any(
            'transforms' in source(call.func) or
            'augment' in source(call.func).lower() or
            'albumentations' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        # Only suggest augmentation for vision tasks during training
        if is_vision_task and has_training and not has_augmentation:
            self.add_smell('PyTorch', 'Data Augmentation Checker', node, file_path)

    @staticmethod
    def _longest_epoch_loop(node: nodes.Module) -> int:
        """Largest literal epoch count of a `for epoch in range(...)` loop (range(N), range(1, N), range(EPOCHS))."""
        longest = 0
        for loop in nodes_of(node, nodes.For):
            if 'epoch' not in source(loop.target).lower() or not isinstance(loop.iter, nodes.Call):
                continue
            args = loop.iter.args
            if callee_name(loop.iter) == 'range' and args:
                stop = int_value(args[1] if len(args) > 1 else args[0])
                start = int_value(args[0]) if len(args) > 1 else 0
                if stop is not None and start is not None:
                    longest = max(longest, stop - start)
        return longest

    def detect_pytorch_lr_scheduler(self, node: nodes.Module, file_path: str):
        # More than 10 epochs with an optimizer and no torch.optim.lr_scheduler (or a transformers get_*_schedule*)
        has_scheduler = any(
            'lr_scheduler' in source(call.func) or
            'LRScheduler' in source(call.func) or
            'schedule' in callee_name(call)
            for call in nodes_of(node, nodes.Call)
        )

        if self._longest_epoch_loop(node) > 10 and not has_scheduler and self._has_torch_optimizer(node):
            self.add_smell('PyTorch', 'Learning Rate Scheduler Checker', node, file_path)

    EXPERIMENT_LOGGERS = ('tensorboard', 'summarywriter', 'wandb', 'mlflow', 'neptune', 'comet_ml', 'aim.', 'clearml',
                          'logger', 'logging', 'csvlogger', 'dvclive')

    def detect_pytorch_logging(self, node: nodes.Module, file_path: str):
        # Check if there's actual training to log
        has_training_loop = any(
            callee_name(call) in ('train', 'backward')
            for call in nodes_of(node, nodes.Call)
        )

        text = source(node).lower()
        has_metrics = any(metric in text for metric in ['loss', 'accuracy', 'score', 'metric'])
        has_logging = any(logger in text for logger in self.EXPERIMENT_LOGGERS)

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
            indicator in source(node).lower()
            for indicator in evaluation_indicators
        )

        has_model_usage = any(
            'forward' in source(call.func) or
            'model(' in source(call)
            for call in nodes_of(node, nodes.Call)
        )

        # model.eval(), or model.train(False) which is the same thing
        has_eval_mode = any(
            callee_name(call) == 'eval' or
            callee_name(call) == 'train' and call.args and isinstance(call.args[0], nodes.Const) and call.args[0].value is False
            for call in nodes_of(node, nodes.Call)
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
        run_checks((
            self.detect_tf_random_seed,
            self.detect_tf_early_stopping,
            self.detect_tf_checkpointing,
            self.detect_tf_memory_release,
            self.detect_tf_mask,
            self.detect_tf_tensor_array,
            self.detect_tf_metrics,
            self.detect_tf_logging,
            self.detect_tf_batch_norm,
            self.detect_tf_dropout,
            self.detect_tf_augmentation,
            self.detect_tf_lr_scheduler,
            self.detect_tf_model_evaluation,
            self.detect_tf_unit_tests,
            self.detect_tf_exception_handling,
        ), node, file_path)

    def detect_tf_random_seed(self, node: nodes.Module, file_path: str):
        # Check for operations that need random seed control
        random_ops = [
            'random.normal', 'random.uniform', 'random.shuffle', 'random.categorical',
            'dropout', 'Dropout', 'RandomRotation', 'RandomFlip', 'RandomZoom', 'RandomCrop'
        ]

        has_random_ops = any(
            op in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for op in random_ops
        )

        has_seed_set = any(
            'random.set_seed' in source(call.func) or
            'set_random_seed' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        # Only report if random operations are used without seed
        if has_random_ops and not has_seed_set:
            self.add_smell('TensorFlow', 'Randomness Control Checker', node, file_path)

    @staticmethod
    def _fit_calls(node: nodes.Module) -> List[nodes.Call]:
        return [c for c in nodes_of(node, nodes.Call) if callee_name(c) == 'fit']

    def detect_tf_early_stopping(self, node: nodes.Module, file_path: str):
        # x.fit(..., epochs=N) with N > 1 (a literal or a name bound to one) and no EarlyStopping callback
        has_early_stopping = any('EarlyStopping' in source(call.func) for call in nodes_of(node, nodes.Call))
        if has_early_stopping:
            return
        for call in self._fit_calls(node):
            epochs = int_value(keyword(call, 'epochs'))
            if epochs is not None and epochs > 1:
                self.add_smell('TensorFlow', 'Early Stopping Checker', call, file_path)
                return

    def detect_tf_checkpointing(self, node: nodes.Module, file_path: str):
        has_complex_model = any(
            layer in source(node)
            for layer in ['Dense', 'Conv', 'LSTM', 'GRU']
        )

        has_checkpointing = any(
            any(ckpt in source(call.func) for ckpt in ('ModelCheckpoint', 'BackupAndRestore', 'Checkpoint')) or
            callee_name(call) in ('save_weights', 'save')
            for call in nodes_of(node, nodes.Call)
        )

        fits = self._fit_calls(node)
        if fits and has_complex_model and not has_checkpointing:
            self.add_smell('TensorFlow', 'Checkpointing Checker', fits[0], file_path)

    def detect_tf_memory_release(self, node: nodes.Module, file_path: str):
        # Models trained inside a loop (hyper-parameter sweeps, k-fold) pile up in the Keras session unless
        # clear_session() is called. A single model per script does not leak, and a loop that only stacks layers
        # (keras.Sequential blocks inside a model definition) builds one model, so the loop must compile or fit.
        if any(callee_name(c) == 'clear_session' for c in nodes_of(node, nodes.Call)):
            return
        reported = []
        for call in nodes_of(node, nodes.Call):
            if callee_name(call) in ('fit', 'compile'):
                loop = in_loop(call)
                if loop is not None and not any(loop is r for r in reported):
                    reported.append(loop)
                    self.add_smell('TensorFlow', 'Memory Release Checker', loop, file_path)

    TF_LOGS = {'tensorflow.math.log', 'tensorflow.log', 'tensorflow.math.log2', 'tensorflow.experimental.numpy.log'}

    def detect_tf_mask(self, node: nodes.Module, file_path: str):
        # tf.math.log(x) of an input that can be zero/negative, with no where/clip/maximum in the module
        import_map = get_import_map(node)
        if any(mask_op in source(node) for mask_op in ['where', 'clip', 'maximum']):
            return
        risky_ops = ['zeros', 'random', 'subtract', 'sub']
        for call in nodes_of(node, nodes.Call):
            if resolve_call_name(call, import_map) in self.TF_LOGS and len(call.args) == 1:
                if any(op in source(call.args[0]) for op in risky_ops):
                    self.add_smell('TensorFlow', 'Mask Missing Checker', call, file_path)

    def detect_tf_tensor_array(self, node: nodes.Module, file_path: str):
        # A Python list grown inside a loop in a @tf.function: it is unrolled at trace time; tf.TensorArray is not
        for func in nodes_of(node, nodes.FunctionDef):
            decorators = func.decorators.nodes if func.decorators else []
            if not any(source(d).split('(')[0].endswith('function') for d in decorators):
                continue
            if any('TensorArray' in source(c.func) for c in nodes_of(func, nodes.Call)):
                continue
            for call in nodes_of(func, nodes.Call):
                if callee_name(call) in ('append', 'extend') and in_loop(call):
                    self.add_smell('TensorFlow', 'Tensor Array Checker', call, file_path)

    def detect_tf_metrics(self, node: nodes.Module, file_path: str):
        # Check if it's a classification task
        classification_indicators = [
            'Binary', 'Categorical', 'sparse_categorical',
            'accuracy', 'precision', 'recall', 'f1'
        ]

        is_classification = any(
            indicator in source(node)
            for indicator in classification_indicators
        )

        # metrics=['accuracy'] is a string, not a call: look at the compile() metrics too
        has_basic_metrics = any(
            metric in source(call.func) or
            callee_name(call) == 'compile' and metric in source(keyword(call, 'metrics') or call)
            for call in nodes_of(node, nodes.Call)
            for metric in ['accuracy', 'precision', 'recall']
        )

        has_threshold_independent = any(
            metric in source(call)
            for call in nodes_of(node, nodes.Call)
            for metric in ['AUC', 'auc', 'AUROCScore', 'PrecisionRecallCurve']
        )

        # Only suggest threshold-independent metrics for classification tasks
        if is_classification and has_basic_metrics and not has_threshold_independent:
            self.add_smell('TensorFlow', 'Dependent Threshold Checker', node, file_path)

    def detect_tf_logging(self, node: nodes.Module, file_path: str):
        # Check if there's actual training to log
        has_training_loop = bool(self._fit_calls(node)) or any(
            'train' in callee_name(call) for call in nodes_of(node, nodes.Call))

        text = source(node).lower()
        has_metrics = any(metric in text for metric in ['loss', 'accuracy', 'score', 'metric'])
        has_logging = any(logger in text for logger in ('tf.summary', 'tensorflow.summary') + self.EXPERIMENT_LOGGERS)

        # Only suggest logging for training with metrics
        if has_training_loop and has_metrics and not has_logging:
            self.add_smell('TensorFlow', 'Logging Checker', node, file_path)

    def detect_tf_batch_norm(self, node: nodes.Module, file_path: str):
        # Check if it's a deep network that could benefit from BatchNorm
        conv_layers = [
            'Conv1D', 'Conv2D', 'Conv3D',
            'Dense', 'SeparableConv'
        ]

        # Check if model is deep enough
        layer_count = len([
            call for call in nodes_of(node, nodes.Call)
            if any(layer in source(call.func)
                   for layer in conv_layers)
        ])

        # Any normalisation layer counts: BatchNormalization, LayerNormalization, GroupNormalization
        has_batch_norm = any(
            'Normalization' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        # Only suggest BatchNorm for deep networks with conv layers
        if layer_count > 2 and not has_batch_norm:
            self.add_smell('TensorFlow', 'Batch Normalisation Checker', node, file_path)

    def detect_tf_dropout(self, node: nodes.Module, file_path: str):
        # Check if model is complex enough to need dropout
        deep_layers = ['Dense', 'Conv', 'LSTM', 'GRU']

        layer_count = len([
            call for call in nodes_of(node, nodes.Call)
            if any(layer in source(call.func)
                   for layer in deep_layers)
        ])

        has_training = bool(self._fit_calls(node)) or any(
            'training=True' in source(call)
            for call in nodes_of(node, nodes.Call)
        )

        has_dropout = any(
            'Dropout' in source(call.func)
            for call in nodes_of(node, nodes.Call)
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
            indicator in source(node)
            for indicator in vision_indicators
        )

        has_training = bool(self._fit_calls(node)) or any(
            'train' in source(call.func).lower()
            for call in nodes_of(node, nodes.Call)
        )

        has_augmentation = any(
            aug in source(call.func)
            for call in nodes_of(node, nodes.Call)
            for aug in ['ImageDataGenerator', 'RandomFlip', 'RandomRotation', 'RandomZoom', 'RandomCrop',
                        'RandomTranslation', 'RandomContrast', 'image.random_', 'image.flip_']
        )

        # Only suggest augmentation for vision tasks during training
        if is_vision_task and has_training and not has_augmentation:
            self.add_smell('TensorFlow', 'Data Augmentation Checker', node, file_path)

    def detect_tf_lr_scheduler(self, node: nodes.Module, file_path: str):
        # A long training run: an epoch loop over more than 5 epochs, or fit(epochs=N) with N > 5
        long_fit = any((int_value(keyword(c, 'epochs')) or 0) > 5 for c in self._fit_calls(node))

        has_optimizer = any(
            'optimizer' in source(call.func).lower()
            for call in nodes_of(node, nodes.Call)
        ) or any(kw.arg == 'optimizer' for call in nodes_of(node, nodes.Call) for kw in call.keywords)

        has_scheduler = any(
            'LearningRateScheduler' in source(call.func) or
            'schedules' in source(call.func) or
            'ReduceLROnPlateau' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        # Only suggest scheduler for long training processes
        if (self._longest_epoch_loop(node) > 5 or long_fit) and has_optimizer and not has_scheduler:
            self.add_smell('TensorFlow', 'Learning Rate Scheduler Checker', node, file_path)

    def detect_tf_model_evaluation(self, node: nodes.Module, file_path: str):
        # Check if there's a model to evaluate
        has_model = any(
            'model' in source(call.func) or
            'Sequential' in source(call.func) or
            'Model(' in source(call.func)
            for call in nodes_of(node, nodes.Call)
        )

        has_test_data = any(
            data in source(node).lower()
            for data in ['test', 'val', 'valid', 'evaluation']
        )

        has_evaluation = any(
            'evaluate' in source(call.func) or
            'predict' in source(call.func)
            for call in nodes_of(node, nodes.Call)
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
        return {"General": [{"name": "Import Checker",
                             "how_to_fix": "Use standard naming conventions for imported modules.",
                             "benefits": "Improves code readability and maintainability.",
                             "strategies": "Follow standard naming conventions (e.g., import numpy as np, import pandas as pd)."}],
                "Pandas": [{"name": "Unnecessary Iteration",
                            "how_to_fix": "Use vectorized operations instead of loops.",
                            "benefits": "Enhances performance and reduces execution time.",
                            "strategies": "Replace loops with Pandas vectorized functions (e.g., apply, map, vectorized arithmetic operations)."},
                           {"name": "DataFrame Iteration Modification",
                            "how_to_fix": "Avoid modifying DataFrame during iteration.",
                            "benefits": "Prevents unexpected behaviour and potential data corruption.",
                            "strategies": "Use temporary variables or vectorized operations for modifications."},
                           {"name": "Chain Indexing",
                            "how_to_fix": "Use single indexing or .loc[], .iloc[] methods.",
                            "benefits": "Enhances code readability and prevents performance issues.",
                            "strategies": "Use .loc[] or .iloc[] for DataFrame indexing instead of chained indexing."},
                           {"name": "Datatype Checker",
                            "how_to_fix": "Set data types explicitly when importing data.",
                            "benefits": "Ensures correct data format and reduces memory usage.",
                            "strategies": "Use dtype parameter in Pandas read functions (e.g., pd.read_csv)."},
                           {"name": "Column Selection Checker",
                            "how_to_fix": "Select necessary columns after importing DataFrame.",
                            "benefits": "Clarifies data usage and improves performance.",
                            "strategies": "Use column selection methods (e.g., df[['col1', 'col2']])."},
                           {"name": "Merge Parameter Checker",
                            "how_to_fix": "Specify how, on, and validate parameters in merge operations.",
                            "benefits": "Ensures accurate data merging and prevents data loss.",
                            "strategies": "Use appropriate parameters in pd.merge function."},
                           {"name": "InPlace Checker",
                            "how_to_fix": "Assign operations to a new DataFrame variable.",
                            "benefits": "Prevents data loss and improves code clarity.",
                            "strategies": "Assign operation results to a new variable instead of using inplace=True."},
                           {"name": "DataFrame Conversion Checker",
                            "how_to_fix": "Use .to_numpy() instead of .values.",
                            "benefits": "Ensures future compatibility and avoids unexpected behaviour.",
                            "strategies": "Replace .values with .to_numpy() in Pandas DataFrame conversion."}],
                "NumPy": [{"name": "NaN Equality Checker",
                           "how_to_fix": "Use np.isnan() to check for NaN values.",
                           "benefits": "Ensures accurate data handling and avoids logical errors.",
                           "strategies": "Replace == np.nan with np.isnan()."},
                          {"name": "Randomness Control Checker",
                           "how_to_fix": "Use np.random.seed() for reproducibility.",
                           "benefits": "Enables reproducible results and debugging.",
                           "strategies": "Set random seed using np.random.seed(seed_value)."},
                          {"name": "Array Creation Efficiency",
                           "how_to_fix": "Specify dtype when creating arrays.",
                           "benefits": "Improves memory efficiency and avoids unnecessary conversions.",
                           "strategies": "Use dtype parameter in np.array, np.zeros, np.ones, np.empty functions."},
                          {"name": "Inefficient Operations",
                           "how_to_fix": "Optimize numerical operations.",
                           "benefits": "Improves performance and reduces execution time.",
                           "strategies": "Use vectorized functions instead of loops for element-wise operations."},
                          {"name": "Dtype Consistency",
                           "how_to_fix": "Ensure consistent data types in operations.",
                           "benefits": "Improves accuracy and avoids logical errors.",
                           "strategies": "Use consistent data types in np.sum, np.mean, np.max, np.min functions."},
                          {"name": "Broadcasting Risk",
                           "how_to_fix": "Specify axis in operations between arrays.",
                           "benefits": "Improves performance and avoids broadcasting issues.",
                           "strategies": "Use axis parameter in np.reshape, np.transpose functions."},
                          {"name": "Copy-View Confusion",
                           "how_to_fix": "Explicitly copy arrays before slicing.",
                           "benefits": "Improves performance and avoids unexpected behaviour.",
                           "strategies": "Use .copy() method or np.copy function before slicing."},
                          {"name": "Missing Axis Specification",
                           "how_to_fix": "Specify axis in array operations.",
                           "benefits": "Improves performance and avoids unexpected behaviour.",
                           "strategies": "Use axis parameter in np.sum, np.mean, np.max, np.min, np.argmax, np.argmin, np.any, np.all functions."}],
                "ScikitLearn": [{"name": "Scaler Missing Checker",
                                 "how_to_fix": "Apply scaling before scaling-sensitive operations.",
                                 "benefits": "Improves model performance and accuracy.",
                                 "strategies": "Use StandardScaler, MinMaxScaler, etc., before applying PCA, SVM, etc."},
                                {"name": "Pipeline Checker",
                                 "how_to_fix": "Use Pipelines for all scikit-learn estimators.",
                                 "benefits": "Prevents data leakage and ensures correct model evaluation.",
                                 "strategies": "Implement Pipeline from sklearn.pipeline for preprocessing and model fitting."},
                                {"name": "Cross Validation Checker",
                                 "how_to_fix": "Implement cross-validation techniques for robust model evaluation.",
                                 "benefits": "Enhances model performance and reduces overfitting.",
                                 "strategies": "Use cross_val_score, KFold, or other cross-validation methods."},
                                {"name": "Randomness Control Checker",
                                 "how_to_fix": "Set random_state in estimators for reproducibility.",
                                 "benefits": "Ensures reproducible results and consistent model behaviour.",
                                 "strategies": "Set random_state to a fixed value in scikit-learn estimators."},
                                {"name": "Verbose Mode Checker",
                                 "how_to_fix": "Enable verbose mode for long training processes.",
                                 "benefits": "Provides better insights into model training progress.",
                                 "strategies": "Pass verbose=1 to search (GridSearchCV, ...) and ensemble constructors; fit() takes no verbose."},
                                {"name": "Dependent Threshold Checker",
                                 "how_to_fix": "Use threshold-independent metrics alongside threshold-dependent ones.",
                                 "benefits": "Provides a comprehensive evaluation of model performance.",
                                 "strategies": "Include metrics like ROC AUC score alongside accuracy or F1-score."},
                                {"name": "Unit Testing Checker",
                                 "how_to_fix": "Write unit tests for data processing and model components.",
                                 "benefits": "Ensures code reliability and prevents bugs.",
                                 "strategies": "Use unittest or pytest to write and run tests for individual components."},
                                {"name": "Data Leakage Checker",
                                 "how_to_fix": "Split data before any preprocessing or feature engineering.",
                                 "benefits": "Prevents data leakage and ensures valid model evaluation.",
                                 "strategies": "Use train_test_split before any data preprocessing steps."},
                                {"name": "Exception Handling Checker",
                                 "how_to_fix": "Handle failures when loading or saving model artifacts.",
                                 "benefits": "Improves code robustness and error reporting.",
                                 "strategies": "Catch specific exceptions (e.g. FileNotFoundError, EOFError) around joblib/pickle load and dump; "
                                               "do not wrap fit/predict in blanket try-except."}],
                "PyTorch": [{"name": "Randomness Control Checker",
                             "how_to_fix": "Set random seed using torch.manual_seed() for reproducibility.",
                             "benefits": "Ensures reproducible results across different runs.",
                             "strategies": "Add torch.manual_seed(seed_value) at the start of your script."},
                            {"name": "Deterministic Algorithm Usage Checker",
                             "how_to_fix": "Enable deterministic algorithms using torch.use_deterministic_algorithms(True).",
                             "benefits": "Ensures consistent results across different hardware and runs.",
                             "strategies": "Set torch.use_deterministic_algorithms(True) and handle any required adjustments."},
                            {"name": "Randomness Control Checker (PyTorch-Dataloader)",
                             "how_to_fix": "Set worker_init_fn and generator in DataLoader for reproducible data loading.",
                             "benefits": "Ensures consistent data loading across different runs.",
                             "strategies": "Configure worker_init_fn and generator parameters in DataLoader initialization."},
                            {"name": "Mask Missing Checker",
                             "how_to_fix": "Use appropriate masking when dealing with log operations.",
                             "benefits": "Prevents numerical errors and improves model stability.",
                             "strategies": "Apply masks before log operations to handle invalid values."},
                            {"name": "Net Forward Checker",
                             "how_to_fix": "Use model(input) instead of model.forward(input).",
                             "benefits": "Follows PyTorch best practices and handles hooks properly.",
                             "strategies": "Replace direct forward() calls with the recommended calling syntax."},
                            {"name": "Gradient Clear Checker",
                             "how_to_fix": "Clear gradients before each backward pass using optimizer.zero_grad().",
                             "benefits": "Prevents gradient accumulation and ensures correct updates.",
                             "strategies": "Add optimizer.zero_grad() before loss.backward() in training loop."},
                            {"name": "Batch Normalisation Checker",
                             "how_to_fix": "Include BatchNorm layers in your model architecture.",
                             "benefits": "Improves training stability and model convergence.",
                             "strategies": "Add torch.nn.BatchNorm layers after convolutional or linear layers."},
                            {"name": "Dropout Usage Checker",
                             "how_to_fix": "Include Dropout layers for regularization.",
                             "benefits": "Reduces overfitting and improves model generalization.",
                             "strategies": "Add torch.nn.Dropout layers in your model architecture."},
                            {"name": "Data Augmentation Checker",
                             "how_to_fix": "Implement data augmentation using torchvision.transforms.",
                             "benefits": "Improves model robustness and reduces overfitting.",
                             "strategies": "Use torchvision.transforms to apply various augmentation techniques."},
                            {"name": "Learning Rate Scheduler Checker",
                             "how_to_fix": "Implement learning rate scheduling.",
                             "benefits": "Improves training convergence and final model performance.",
                             "strategies": "Use torch.optim.lr_scheduler for dynamic learning rate adjustment."},
                            {"name": "Logging Checker",
                             "how_to_fix": "Implement proper logging using tensorboard or similar tools.",
                             "benefits": "Enables better experiment tracking and debugging.",
                             "strategies": "Use tensorboardX or torch.utils.tensorboard for logging."},
                            {"name": "Model Evaluation Checker",
                             "how_to_fix": "Set model to evaluation mode using model.eval().",
                             "benefits": "Ensures correct behavior of layers like BatchNorm and Dropout during inference.",
                             "strategies": "Call model.eval() before validation/testing and model.train() before training."}],
                "TensorFlow": [{"name": "Randomness Control Checker",
                                "how_to_fix": "Set random seed using tf.random.set_seed().",
                                "benefits": "Ensures reproducible results across different runs.",
                                "strategies": "Add tf.random.set_seed(seed_value) at the start of your script."},
                               {"name": "Early Stopping Checker",
                                "how_to_fix": "Implement early stopping using tf.keras.callbacks.EarlyStopping.",
                                "benefits": "Prevents overfitting and reduces unnecessary training time.",
                                "strategies": "Add EarlyStopping callback to model.fit()."},
                               {"name": "Checkpointing Checker",
                                "how_to_fix": "Implement model checkpointing using tf.keras.callbacks.ModelCheckpoint.",
                                "benefits": "Enables model recovery and saves best performing models.",
                                "strategies": "Add ModelCheckpoint callback to model.fit()."},
                               {"name": "Memory Release Checker",
                                "how_to_fix": "Clear session after model creation/training using tf.keras.backend.clear_session().",
                                "benefits": "Prevents memory leaks and reduces resource usage.",
                                "strategies": "Call clear_session() after completing major operations."},
                               {"name": "Mask Missing Checker",
                                "how_to_fix": "Use appropriate masking for log operations.",
                                "benefits": "Prevents numerical errors and improves model stability.",
                                "strategies": "Apply masks before log operations to handle invalid values."},
                               {"name": "Tensor Array Checker",
                                "how_to_fix": "Use tf.TensorArray for dynamic tensor operations.",
                                "benefits": "Improves memory efficiency for dynamic computations.",
                                "strategies": "Replace Python lists with tf.TensorArray for growing tensors."},
                               {"name": "Dependent Threshold Checker",
                                "how_to_fix": "Use threshold-independent metrics like AUC.",
                                "benefits": "Provides more robust model evaluation.",
                                "strategies": "Include tf.keras.metrics.AUC in your model metrics."},
                               {"name": "Logging Checker",
                                "how_to_fix": "Implement TensorBoard logging.",
                                "benefits": "Enables better experiment tracking and visualization.",
                                "strategies": "Use tf.summary or TensorBoard callbacks for logging."},
                               {"name": "Batch Normalisation Checker",
                                "how_to_fix": "Include BatchNormalization layers in your model.",
                                "benefits": "Improves training stability and model convergence.",
                                "strategies": "Add tf.keras.layers.BatchNormalization layers to your model."},
                               {"name": "Dropout Usage Checker",
                                "how_to_fix": "Include Dropout layers for regularization.",
                                "benefits": "Reduces overfitting and improves generalization.",
                                "strategies": "Add tf.keras.layers.Dropout layers to your model."},
                               {"name": "Data Augmentation Checker",
                                "how_to_fix": "Implement data augmentation using ImageDataGenerator.",
                                "benefits": "Improves model robustness and reduces overfitting.",
                                "strategies": "Use tf.keras.preprocessing.image.ImageDataGenerator for augmentation."},
                               {"name": "Learning Rate Scheduler Checker",
                                "how_to_fix": "Implement learning rate scheduling.",
                                "benefits": "Improves training convergence and final model performance.",
                                "strategies": "Use LearningRateScheduler callback or custom scheduling."},
                               {"name": "Model Evaluation Checker",
                                "how_to_fix": "Evaluate model performance using model.evaluate().",
                                "benefits": "Provides standardized model evaluation.",
                                "strategies": "Use model.evaluate() on validation/test data."},
                               {"name": "Unit Testing Checker",
                                "how_to_fix": "Implement unit tests using tf.test.TestCase.",
                                "benefits": "Ensures code reliability and prevents regressions.",
                                "strategies": "Create test classes inheriting from tf.test.TestCase."},
                               {"name": "Exception Handling Checker",
                                "how_to_fix": "Implement proper exception handling.",
                                "benefits": "Improves code robustness and error reporting.",
                                "strategies": "Use try-except blocks to handle TensorFlow-specific exceptions."}]}
