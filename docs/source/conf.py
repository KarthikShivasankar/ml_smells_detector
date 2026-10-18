import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

project = 'ML Code Smell Detector'
copyright = '2026, Karthik Shivashankar'
author = 'Karthik Shivashankar'
# Single source of truth: the version in pyproject.toml (tests/test_release.py checks the rest agree)
with open(os.path.join(os.path.dirname(__file__), '../../pyproject.toml'), encoding='utf-8') as _f:
    release = re.search(r'^version = "([^"]+)"', _f.read(), re.M).group(1)
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'