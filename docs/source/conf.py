# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Merlin'
copyright = '2022, Nguyen Dinh Quoc Dang'
author = 'Nguyen Dinh Quoc Dang'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'breathe',
    'sphinx_doxysummary'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Doxygen generated XML files
breathe_projects = { "merlin": os.path.abspath("./xml") }
breathe_default_project = "merlin"
doxygen_xml = [breathe_projects[breathe_default_project]]


# -- Pygments style ----------------------------------------------------------

pygments_style = 'rainbow_dash'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_css_files = ['azura.css']
html_js_files = [('azura.js', {'defer': 'defer'})]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Do not add an index to the HTML documents.
html_use_index = False


# -- Options for Latex output ------------------------------------------------

# latex_theme = 'howto'
latex_engine = 'xelatex'
latex_additional_files = ['_static/azura.sty',  # '_static/logo.pdf',
                          '_static/azura.pdf', '_static/LeagueSpartan.otf',
                          '_static/FiraCode.ttf', '_static/FiraCode-SemiBold.ttf',
                          '_static/ChunkFive.ttf', '_static/CocoGoose.ttf']
latex_elements = {
    'papersize': 'a4paper',
    'passoptionstopackages': r'\PassOptionsToPackage{explicit}{titlesec}',
    'fontpkg': '',
    'fncychap': '',
    'figure_align': 'htbp',
    'pointsize': '10pt',
    'tableofcontents': ('\\renewcommand{\\contentsname}{Contents}\n'
                        r'\tableofcontents\clearpage\pagenumbering{arabic}'),
    'preamble': r'\usepackage{azura}',
    # 'makeindex': r'\usepackage[columns=1]{idxlayout}\makeindex',
    'makeindex': '',
}
