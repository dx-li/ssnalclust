"""Sphinx configuration for the installed ssnalclust package."""

from importlib.metadata import version as package_version

project = "ssnalclust"
author = "ssnalclust contributors"
copyright = "2026, dx-li"
release = package_version("ssnalclust")
version = release
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 4
nitpicky = True
autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
html_theme = "alabaster"
html_title = "ssnalclust — development documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "description": "Convex clustering with numerical certificates",
    "fixed_sidebar": True,
    "page_width": "1160px",
    "sidebar_width": "250px",
}
html_sidebars = {"**": ["about.html", "searchbox.html", "navigation.html"]}
