from __future__ import annotations

import importlib.metadata as metadata
import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLUGINS", "disabled")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
EXT_DIR = Path(__file__).resolve().parent / "_ext"
if EXT_DIR.exists():
    sys.path.insert(0, str(EXT_DIR))

# -- Project information -----------------------------------------------------

project = "OMpy"
author = "OMpy contributors"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

try:
    release = metadata.version("ompy")
except metadata.PackageNotFoundError:
    release = "0.0.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "jupyter_execute",
    # Uncommented temporarily to speed up build
    #"autoapi.extension",
    #"sphinx.ext.autodoc",
    #"sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"
default_role = "py:obj"

autosummary_generate = True
autosummary_imported_members = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "numba",
    "cupy",
    "optax",
    "uproot",
]
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
todo_include_todos = True
nbsphinx_execute = "auto"
suppress_warnings = ["ref.intersphinx", "autoapi.python_import_resolution", "ref.python"]

autoapi_type = "python"
autoapi_dirs = [str(SRC_DIR / "ompy")]
autoapi_modules = {"ompy": str(SRC_DIR / "ompy")}
autoapi_keep_files = False
autoapi_root = "reference/autoapi"
autoapi_member_order = "bysource"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_add_toctree_entry = False
autoapi_python_class_content = "class"
autoapi_ignore = [
    "*/landaupy/*",
    "*/landaupy.*",
    "*/tests/*",
    "*/test_*.py",
    "*/test.py",
    "*/setup.py",
    "*/.mypy_cache/*",
    "*/.pytest_cache/*",
    "*/.*/*",
    "*/GPATH",
    "*/GRTAGS",
    "*/GTAGS",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

extlinks = {
    "gh-issue": ("https://github.com/oslocyclotronlab/ompy/issues/%s", "#"),
    "gh-pr": ("https://github.com/oslocyclotronlab/ompy/pull/%s", "PR #"),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_title = "OMpy documentation"
html_theme_options = {
    "logo": {"text": "OMpy"},
    "show_prev_next": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": False,
    "primary_sidebar_end": ["indices.html", "searchbox.html"],
    "header_links_before_dropdown": 4,
    "announcement": (
        "<strong>Development preview:</strong> OMpy 2.0 and its documentation "
        "are actively evolving. Expect breaking changes, missing sections, and "
        "rough edges until the first stable release."
    ),
}

# Allow references to sections across documents without prefix clashes
autosectionlabel_prefix_document = True
