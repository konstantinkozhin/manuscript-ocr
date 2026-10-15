# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import inspect
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# -- Project information -----------------------------------------------------
project = "manuscript-ocr"
copyright = "2026, Konstantin Kozhin"
author = "Konstantin Kozhin"
release = os.environ.get("DOCS_RELEASE", "0.1.13")
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
html_favicon = "_static/favicon.ico"
html_css_files = ["language_switcher.css", "model_zoo.css"]
html_js_files = ["language_switcher.js", "version_sidebar.js"]

# -- Extension configuration -------------------------------------------------

# autodoc settings
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": (
        "__weakref__, default_weights_name, registry_model_class, "
        "pretrained_registry, config_registry, charset_registry, registry"
    ),
    "show-inheritance": True,
}
# Keep constructor documentation under the class's single signature.
autoclass_content = "both"
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
numpydoc_show_class_members = False
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


def data_class_signature(app, what, name, obj, options, signature, return_annotation):
    """Expose public parameters without Pydantic or Pipeline internals."""
    from manuscript.data import Element

    if what == "class" and name == "manuscript.Pipeline":
        return (
            "(detector=<default>, layout=<default>, recognizer=<default>, "
            "corrector=None, layout_after='detector')", None,
        )

    if what == "class" and inspect.isclass(obj) and issubclass(obj, Element):
        model_signature = inspect.signature(obj)
        parameters = [
            parameter.replace(annotation=inspect.Parameter.empty)
            for parameter in model_signature.parameters.values()
        ]
        return str(model_signature.replace(
            parameters=parameters, return_annotation=inspect.Signature.empty,
        )), None


def setup(app):
    app.connect("autodoc-process-signature", data_class_signature, priority=100)
    app.connect("autodoc-process-docstring", data_class_docstring, priority=100)
    app.connect("doctree-resolved", localize_mermaid_labels)


def data_class_docstring(app, what, name, obj, options, lines):
    """Keep data API docs without module or inherited constructor introductions."""
    if what == "module" and name == "manuscript.data":
        lines.clear()
        return

    from manuscript.data import Element

    if what == "class" and inspect.isclass(obj) and issubclass(obj, Element):
        # With autoclass_content="both", Sphinx processes the class and
        # inherited Pydantic constructor docstrings separately.
        if inspect.cleandoc("\n".join(lines)) == inspect.getdoc(obj.__init__):
            lines.clear()
        else:
            lines[:] = inspect.cleandoc(obj.__doc__ or "").splitlines()


def localize_mermaid_labels(app, doctree, docname):
    """Translate diagram prose, which Mermaid does not expose to gettext."""
    if app.config.language != "ru":
        return
    from sphinxcontrib.mermaid import mermaid

    labels = {
        "Registry and artifact cache": "Реестр и кеш артефактов",
        "Base classes and protocols": "Базовые классы и протоколы",
        "type: word / line / block": "тип: word / line / block",
    }
    for node in doctree.findall(mermaid):
        for source, translation in labels.items():
            node["code"] = node["code"].replace(source, translation)
