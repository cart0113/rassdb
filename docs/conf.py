"""Sphinx configuration for RASSDB documentation."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'RASSDB'
copyright = '2025, AJ Carter'
author = 'AJ Carter'
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    "github_url": "https://github.com/ajcarter/rassdb",
    "show_nav_level": 2,
    "navigation_depth": 3,
}

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}