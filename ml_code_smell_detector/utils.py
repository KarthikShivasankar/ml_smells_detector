import os
import astroid

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
    for import_node in node.nodes_of_class((astroid.Import, astroid.ImportFrom)):
        if isinstance(import_node, astroid.Import):
            imported_modules.extend([name for name, _ in import_node.names])
        elif isinstance(import_node, astroid.ImportFrom):
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
    return [func.name for func in node.nodes_of_class(astroid.FunctionDef)]

def get_class_names(node):
    """
    Get a list of class names defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of class names defined in the node.
    """
    return [cls.name for cls in node.nodes_of_class(astroid.ClassDef)]

def get_variable_names(node):
    """
    Get a list of variable names defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of variable names defined in the node.
    """
    return [assign.targets[0].name for assign in node.nodes_of_class(astroid.Assign) if isinstance(assign.targets[0], astroid.AssignName)]

def get_call_names(node):
    """
    Get a list of function/method call names in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of function/method call names in the node.
    """
    return [call.func.as_string() for call in node.nodes_of_class(astroid.Call)]

def get_attribute_names(node):
    """
    Get a list of attribute names accessed in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of attribute names accessed in the node.
    """
    return [attr.attrname for attr in node.nodes_of_class(astroid.Attribute)]

def get_constant_values(node):
    """
    Get a list of constant values defined in an AST node.

    Args:
    node (astroid.Module): The AST node to analyze.

    Returns:
    list: A list of constant values defined in the node.
    """
    return [const.value for const in node.nodes_of_class(astroid.Const)]