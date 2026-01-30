import importlib

# Fail fast if module is not importable
importlib.import_module("fluidgym")

import datetime

project = "FluidGym"
author = "Jannis Becktepe, Safe Autonomous Systems (SAS), TU Dortmund University"
copyright = (
    f"{datetime.date.today().strftime('%Y')}, "
    "Safe Autonomous Systems (SAS), TU Dortmund University"
)
release = "0.1.0"
version = "0.1.0"

templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_logo = "_static/img/logo_lm.png"
html_context = {
    "display_github": True,
    "github_url": "https://github.com/safe-autonomous-systems/fluidgym",
    "github_user": "safe-autonomous-systems",
    "github_repo": "fluidgym",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": True,  
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

autodoc_mock_imports = [
    "fluidgym.simulation.extensions",
    "fluidgym.simulation.pict",
]

autosummary_generate = True
autosummary_generate_overwrite = True