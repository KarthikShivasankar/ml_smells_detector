import functools
import os
import sys
import tokenize

import astroid
from astroid import nodes


def read_source(file_path):
    """Read a Python source file honouring its PEP 263 encoding cookie (UTF-8 by default)."""
    with tokenize.open(file_path) as f:
        return f.read()


def parse_file(file_path):
    """Read and parse a source file. The three detectors analyse the same file, so they share one parse."""
    return _parse(read_source(file_path), os.path.splitext(os.path.basename(file_path))[0])


@functools.lru_cache(maxsize=8)  # keyed on the content itself, so an edited file is never served stale
def _parse(content, module_name):
    return astroid.parse(content, module_name=module_name)


# Checks are read-only over the tree and keep rendering / walking the same nodes (often the whole module), which
# dominated the profile. Nodes hash by identity, and a cached node stays referenced, so keys are never reused.
@functools.lru_cache(maxsize=1 << 16)
def source(node):
    """``node.as_string()``, memoised."""
    return node.as_string()


@functools.lru_cache(maxsize=1 << 12)
def nodes_of(node, klass):
    """``tuple(node.nodes_of_class(klass))``, memoised."""
    return tuple(node.nodes_of_class(klass))


def run_checks(checks, node, file_path):
    """Run each check(node, file_path); a crash in one is reported on stderr and does not skip the rest."""
    for check in checks:
        try:
            check(node, file_path)
        except Exception as e:
            print(f"Error in {check.__name__} for {file_path}: {type(e).__name__}: {e}", file=sys.stderr)


def callee_name(call):
    """Last component of a call's callee: ``a.b.fit(x)`` -> ``'fit'``, ``fit(x)`` -> ``'fit'``, else ``''``."""
    func = call.func
    if isinstance(func, nodes.Attribute):
        return func.attrname
    return func.name if isinstance(func, nodes.Name) else ''


def keyword(call, name):
    """The value node passed as ``name=`` to ``call``, or None."""
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def is_true(value):
    """True for a literal ``True``; a non-literal (``fp16=USE_FP16``) counts as set, since its value is unknown."""
    return value is not None and not (isinstance(value, nodes.Const) and not value.value)


def int_value(node):
    """Integer value of a literal or of a name bound to one (``EPOCHS = 50; range(EPOCHS)``), else None."""
    if node is None:
        return None
    try:
        inferred = next(node.infer())
    except Exception:
        return None
    if isinstance(inferred, nodes.Const) and isinstance(inferred.value, int) and not isinstance(inferred.value, bool):
        return inferred.value
    return None


def in_loop(node):
    """The innermost for/while loop around ``node``, or None."""
    return next((a for a in node.node_ancestors() if isinstance(a, (nodes.For, nodes.While))), None)


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_sample_file(filename, content):
    """
    Create a sample file with the given filename and content.

    Args:
    filename (str): The name of the file to create.
    content (str): The content to write to the file.
    """
    with open(filename, 'w') as f:
        f.write(content)


def get_file_extension(filename):
    """
    Get the extension of a file.

    Args:
    filename (str): The name of the file.

    Returns:
    str: The file extension (without the dot).
    """
    return os.path.splitext(filename)[1][1:]


def is_python_file(filename):
    """
    Check if a file is a Python file based on its extension.

    Args:
    filename (str): The name of the file to check.

    Returns:
    bool: True if the file is a Python file, False otherwise.
    """
    return get_file_extension(filename).lower() == 'py'


def count_lines(filename):
    """
    Count the number of lines in a file.

    Args:
    filename (str): The name of the file to count lines in.

    Returns:
    int: The number of lines in the file.
    """
    with open(filename, 'r') as f:
        return sum(1 for _ in f)


def get_imported_modules(node):
    """
    Get a list of imported modules from an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of imported module names.
    """
    imported_modules = []
    for import_node in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
        if isinstance(import_node, nodes.Import):
            imported_modules.extend([name for name, _ in import_node.names])
        elif isinstance(import_node, nodes.ImportFrom):
            imported_modules.append(import_node.modname)
    return imported_modules


def get_function_names(node):
    """
    Get a list of function names defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of function names defined in the node.
    """
    return [func.name for func in node.nodes_of_class(nodes.FunctionDef)]


def get_class_names(node):
    """
    Get a list of class names defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of class names defined in the node.
    """
    return [cls.name for cls in node.nodes_of_class(nodes.ClassDef)]


def get_variable_names(node):
    """
    Get a list of variable names defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of variable names defined in the node.
    """
    return [
        assign.targets[0].name for assign in node.nodes_of_class(
            nodes.Assign) if isinstance(
            assign.targets[0],
            nodes.AssignName)]


def get_call_names(node):
    """
    Get a list of function/method call names in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of function/method call names in the node.
    """
    return [call.func.as_string() for call in node.nodes_of_class(nodes.Call)]


def get_attribute_names(node):
    """
    Get a list of attribute names accessed in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of attribute names accessed in the node.
    """
    return [attr.attrname for attr in node.nodes_of_class(nodes.Attribute)]


def get_constant_values(node):
    """
    Get a list of constant values defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of constant values defined in the node.
    """
    return [const.value for const in node.nodes_of_class(nodes.Const)]


@functools.lru_cache(maxsize=64)
def get_import_map(node):
    """
    Map each name bound by an import statement to the dotted name it refers to.

    ``import numpy as np`` gives ``{'np': 'numpy'}``; ``from sklearn.utils import shuffle``
    gives ``{'shuffle': 'sklearn.utils.shuffle'}``.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    dict: Local name -> dotted origin. Memoised per module: treat it as read-only.
    """
    imports = {}
    for imp in node.nodes_of_class((nodes.Import, nodes.ImportFrom)):
        for name, alias in imp.names:
            if isinstance(imp, nodes.ImportFrom):
                imports[alias or name] = f"{imp.modname}.{name}"
            elif alias:
                imports[alias] = name
            else:
                root = name.split('.')[0]
                imports[root] = root
    return imports


def resolve_call_name(call, import_map):
    """
    Return the dotted origin of a call's callee, e.g. ``np.zeros(3)`` -> ``'numpy.zeros'``.

    Args:
    call (astroid.nodes.Call): The call to resolve.
    import_map (dict): Result of :func:`get_import_map` for the module.

    Returns:
    str or None: The dotted name, or None if the callee is not rooted in an imported name.
    """
    func, parts = call.func, []
    while isinstance(func, nodes.Attribute):
        parts.append(func.attrname)
        func = func.expr
    if not isinstance(func, nodes.Name) or func.name not in import_map:
        return None
    return '.'.join([import_map[func.name]] + parts[::-1])
