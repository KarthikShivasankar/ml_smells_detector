"""Tests for utility functions in ml_code_smell_detector/utils.py."""

import os

import astroid
import pytest

from ml_code_smell_detector.utils import (
    count_lines,
    create_sample_file,
    ensure_directory_exists,
    get_attribute_names,
    get_call_names,
    get_class_names,
    get_constant_values,
    get_file_extension,
    get_function_names,
    get_imported_modules,
    get_variable_names,
    is_python_file,
)

# ---------------------------------------------------------------------------
# ensure_directory_exists
# ---------------------------------------------------------------------------


class TestEnsureDirectoryExists:
    def test_creates_new_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_subdir")
        assert not os.path.exists(new_dir)
        ensure_directory_exists(new_dir)
        assert os.path.isdir(new_dir)

    def test_does_not_fail_if_directory_exists(self, tmp_path):
        existing = str(tmp_path)
        ensure_directory_exists(existing)  # should not raise
        assert os.path.isdir(existing)


# ---------------------------------------------------------------------------
# create_sample_file
# ---------------------------------------------------------------------------

class TestCreateSampleFile:
    def test_creates_file_with_content(self, tmp_path):
        path = str(tmp_path / "sample.py")
        create_sample_file(path, "x = 1\n")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "x = 1\n"

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "sample.py")
        create_sample_file(path, "old content")
        create_sample_file(path, "new content")
        with open(path) as f:
            assert f.read() == "new content"


# ---------------------------------------------------------------------------
# get_file_extension
# ---------------------------------------------------------------------------

class TestGetFileExtension:
    def test_python_file(self):
        assert get_file_extension("script.py") == "py"

    def test_csv_file(self):
        assert get_file_extension("data.csv") == "csv"

    def test_file_with_path(self):
        assert get_file_extension("/home/user/model.pkl") == "pkl"

    def test_multiple_dots(self):
        assert get_file_extension("archive.tar.gz") == "gz"

    def test_no_extension(self):
        assert get_file_extension("Makefile") == ""


# ---------------------------------------------------------------------------
# is_python_file
# ---------------------------------------------------------------------------

class TestIsPythonFile:
    def test_py_extension(self):
        assert is_python_file("script.py") is True

    def test_PY_uppercase(self):
        assert is_python_file("script.PY") is True

    def test_not_python(self):
        assert is_python_file("data.csv") is False

    def test_txt_file(self):
        assert is_python_file("readme.txt") is False


# ---------------------------------------------------------------------------
# count_lines
# ---------------------------------------------------------------------------

class TestCountLines:
    def test_counts_lines_correctly(self, tmp_path):
        p = tmp_path / "f.py"
        p.write_text("line1\nline2\nline3\n")
        assert count_lines(str(p)) == 3

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.py"
        p.write_text("")
        assert count_lines(str(p)) == 0

    def test_single_line_no_newline(self, tmp_path):
        p = tmp_path / "one.py"
        p.write_text("x = 1")
        assert count_lines(str(p)) == 1


# ---------------------------------------------------------------------------
# AST helpers — parse once, reuse
# ---------------------------------------------------------------------------

SAMPLE_CODE = """\
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

MY_CONSTANT = 42
name = "hello"

def train_model(X, y):
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

class MyModel:
    def predict(self, X):
        pass
"""


@pytest.fixture
def parsed_module():
    return astroid.parse(SAMPLE_CODE, module_name="sample")


# ---------------------------------------------------------------------------
# get_imported_modules
# ---------------------------------------------------------------------------

class TestGetImportedModules:
    def test_finds_top_level_imports(self, parsed_module):
        modules = get_imported_modules(parsed_module)
        assert "os" in modules
        assert "numpy" in modules

    def test_finds_from_imports(self, parsed_module):
        modules = get_imported_modules(parsed_module)
        assert "sklearn.linear_model" in modules

    def test_returns_list(self, parsed_module):
        result = get_imported_modules(parsed_module)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_function_names
# ---------------------------------------------------------------------------

class TestGetFunctionNames:
    def test_finds_function(self, parsed_module):
        names = get_function_names(parsed_module)
        assert "train_model" in names

    def test_finds_method(self, parsed_module):
        names = get_function_names(parsed_module)
        assert "predict" in names

    def test_returns_list(self, parsed_module):
        assert isinstance(get_function_names(parsed_module), list)


# ---------------------------------------------------------------------------
# get_class_names
# ---------------------------------------------------------------------------

class TestGetClassNames:
    def test_finds_class(self, parsed_module):
        names = get_class_names(parsed_module)
        assert "MyModel" in names

    def test_returns_list(self, parsed_module):
        assert isinstance(get_class_names(parsed_module), list)

    def test_empty_for_no_classes(self):
        module = astroid.parse("x = 1", module_name="noclass")
        assert get_class_names(module) == []


# ---------------------------------------------------------------------------
# get_variable_names
# ---------------------------------------------------------------------------

class TestGetVariableNames:
    def test_finds_variable(self, parsed_module):
        names = get_variable_names(parsed_module)
        assert "MY_CONSTANT" in names

    def test_finds_string_variable(self, parsed_module):
        names = get_variable_names(parsed_module)
        assert "name" in names

    def test_returns_list(self, parsed_module):
        assert isinstance(get_variable_names(parsed_module), list)


# ---------------------------------------------------------------------------
# get_call_names
# ---------------------------------------------------------------------------

class TestGetCallNames:
    def test_finds_constructor_call(self, parsed_module):
        calls = get_call_names(parsed_module)
        assert "LogisticRegression" in calls

    def test_finds_method_call(self, parsed_module):
        calls = get_call_names(parsed_module)
        assert any("fit" in c for c in calls)

    def test_returns_list(self, parsed_module):
        assert isinstance(get_call_names(parsed_module), list)


# ---------------------------------------------------------------------------
# get_attribute_names
# ---------------------------------------------------------------------------

class TestGetAttributeNames:
    def test_finds_fit_attribute(self, parsed_module):
        attrs = get_attribute_names(parsed_module)
        assert "fit" in attrs

    def test_returns_list(self, parsed_module):
        assert isinstance(get_attribute_names(parsed_module), list)


# ---------------------------------------------------------------------------
# get_constant_values
# ---------------------------------------------------------------------------

class TestGetConstantValues:
    def test_finds_integer_constant(self, parsed_module):
        consts = get_constant_values(parsed_module)
        assert 42 in consts

    def test_finds_string_constant(self, parsed_module):
        consts = get_constant_values(parsed_module)
        assert "hello" in consts

    def test_returns_list(self, parsed_module):
        assert isinstance(get_constant_values(parsed_module), list)
