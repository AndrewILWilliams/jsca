"""Sphinx configuration for the jsca documentation.

Built on ReadTheDocs from ``docs/`` using MyST-Markdown sources. The site is
authored in Markdown (MyST) so it stays close to the project's existing
``docs/*.md`` design notes, with LaTeX math via ``dollarmath`` and card grids
via ``sphinx-design``.
"""

from __future__ import annotations

import importlib.metadata

# -- Project information -----------------------------------------------------

project = "jsca"
copyright = "2026, the PortingISCA project"
author = "The PortingISCA project"

try:
    release = importlib.metadata.version("jsca")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - docs-only env
    release = "0.0.1"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

# MyST extensions: dollar/amsmath for the governing equations, colon fences and
# card grids for the gallery, deflist for the option tables.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_image",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# Design/roadmap notes that predate the docs site and are not part of the
# published manual. Kept in-repo but excluded from the build so Sphinx does not
# warn about documents outside any toctree.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "scoping.md",
    "frierson_roadmap.md",
    "phase0_checklist.md",
    "held_suarez_timeloop_fidelity.md",
]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# If jax/jsca cannot be imported in the build environment, autodoc still renders
# the rest of the site instead of failing the whole build.
autodoc_mock_imports = []

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "jsca"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/AndrewILWilliams/jsca",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/AndrewILWilliams/jsca",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16"><path fill-rule="evenodd" '
                'd="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
                '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 '
                '2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 '
                '1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 '
                '2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

pygments_style = "friendly"
pygments_dark_style = "monokai"

# -- Source parsing ----------------------------------------------------------

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
