import datetime

# Fail fast if module is not importable
import fluidgym

project = "FluidGym"
author = "Jannis Becktepe, Safe Autonomous Systems (SAS), TU Dortmund University"
copyright = (
    f"{datetime.date.today().strftime('%Y')}, "
    "Safe Autonomous Systems (SAS), TU Dortmund University"
)
release = fluidgym.__version__
version = ".".join(release.split(".")[:2])

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
    "sphinx.ext.intersphinx",
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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Inventories are fetched at build time; keep an offline build from stalling.
intersphinx_timeout = 10

autodoc_mock_imports = [
    "phipict",
]

autosummary_generate = True
autosummary_generate_overwrite = True

autosectionlabel_prefix_document = True
