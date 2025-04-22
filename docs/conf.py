# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add project root to sys.path for autodoc
sys.path.insert(0, os.path.abspath('..'))

project = 'tensorflow-riemopt'
author = 'Oleg Smirnov'
# The full version, including alpha/beta/rc tags
release = '0.3.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
# Exclude Jupyter checkpoints
exclude_patterns = ['**.ipynb_checkpoints']

# HTML output
html_title = "tensorflow-riemopt documentation"
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    "primary_sidebar_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/master/tensorflow-riemopt",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
    "collapse_navigation": True,
}
html_context = {
    "github_user": "master",
    "github_repo": "tensorflow-riemopt",
    "doc_path": "docs",
    "default_mode": "light",
}

# Autodoc settings
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Source suffix and master doc
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
master_doc = 'index'
