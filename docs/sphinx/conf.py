# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "manuscript-ocr"
copyright = "2026, Konstantin Kozhin"
author = "Konstantin Kozhin"
release = os.environ.get("DOCS_RELEASE", "0.1.10")
version = os.environ.get("DOCS_VERSION", release)

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["language_switcher.css"]
html_js_files = ["language_switcher.js"]

# -- Extension configuration -------------------------------------------------

# autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "onnxruntime",
    "cv2",
    "PIL",
    "numpy",
    "shapely",
    "skimage",
    "numba",
    "pydantic",
    "tqdm",
    "gdown",
    "openai",
    "albumentations",
    "scipy",
    "matplotlib",
    "pandas",
    "tensorboard",
]

# Suppress duplicate warnings
suppress_warnings = ["autodoc.import_object"]

# autosummary settings
autosummary_generate = True

# numpydoc settings
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# napoleon settings (backup for docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML theme --------------------------------------------------
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Source suffix
source_suffix = ".rst"

# Master doc
master_doc = "index"

# -- Internationalization (i18n) ---------------------------------------------
locale_dirs = ["locale/"]
gettext_compact = False
gettext_uuid = True
