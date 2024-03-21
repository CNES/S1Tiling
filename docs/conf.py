# -*- coding: utf-8 -*-
#
# S1Tiling documentation build configuration file, created by
# sphinx-quickstart on Mon Sep 14 12:38:14 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
import subprocess
# import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('..'))

BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
metadata = {}
with open(os.path.join(BASEDIR, "s1tiling", "__meta__.py"), "r") as f:
    exec(f.read(), metadata)

def setup(app):
    # app.add_stylesheet("css/otb_theme.css")
    # Customize read the docs theme a bit with a custom css
    # and a custom version selection widget
    # taken from https://stackoverflow.com/a/43186995/5815110
    app.add_js_file("js/versions.js")
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read', ultimateReplace)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx.ext.autosectionlabel",
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    "sphinx_rtd_theme",
    'm2r2'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = metadata['__title__']
copyright = metadata['__copyright__']
author = metadata['__author__']

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = metadata['__version__']
git_version = os.environ.get('CI_COMMIT_REF_NAME') or subprocess.check_output('git describe --tags --exact-match 2> /dev/null || git symbolic-ref -q --short HEAD', shell=True).strip().decode('ascii')
# The full version, including alpha/beta/rc tags.
tag_re = re.compile(r'^\d+\.\d+.*')
match = tag_re.match(git_version)
if match:
    # Releasing a tag
    print('This is a tag: %s'% (git_version,))
    release = git_version
else:
    # Releasing a branch
    print('This is a branch: %s'% (git_version,))
    release = version+'-'+git_version
version = git_version
release_badge = release.replace('-', '--')

print('git_version: %s' % (git_version,))

rst_prolog = """
.. |Badge doc| image:: https://img.shields.io/badge/docs-{release_badge}-brightgreen
   :target: https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/{release}/
""".format(release=release, release_badge=release_badge)


## Enable replacements in code-block and other places
# https://github.com/sphinx-doc/sphinx/issues/4054
def ultimateReplace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result

ultimate_replacements = {
    "{VERSION}" : version
}

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'default'
# html_theme = 'nature'
# html_theme = 'bizstyle'
html_theme = 'sphinx_rtd_theme'
# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()+"/sphinx_rtd_theme/static/"]
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # "show_prev_next": False,
    # 'stickysidebar': True,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Override tables width from RTD theme, and other options
html_css_files = [
    'theme_overrides.css'
]

#
# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'S1Tilingdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'S1Tiling.tex', u'S1Tiling Documentation',
     u'Tierry Koleck, Luc Hermitte', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 's1tiling', u'S1Tiling Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'S1Tiling', u'S1Tiling Documentation',
     author, 'S1Tiling', 'On demand Ortho-rectification of Sentinel-1 data on Sentinel-2 grid.',
     'Miscellaneous'),
]


# Configuration for intersphinx
intersphinx_mapping = {
    "Python": ("https://docs.python.org/3/", None),
    "Distributed": ("https://distributed.dask.org/en/latest/", None),
    "Dask": ("https://docs.dask.org/en/latest/", None),
    "Gdal": ("https://gdal.org/", None),
    # "numpy": ("http://docs.scipy.org/doc/numpy", None),


    # 'https://www.orfeo-toolbox.org/CookBook/': None,
    # Using CookBook from OTB 7.4 as it still distributes DiapOTB.
    "OTB": ('https://www.orfeo-toolbox.org/CookBook-7.4/', None),
}

# Search w/
# $> python -msphinx.ext.intersphinx https://gdal.org/objects.inv
# $> python -msphinx.ext.intersphinx https://www.orfeo-toolbox.org/CookBook/objects.inv
# $> python -msphinx.ext.intersphinx https://github.com/OSGeo/gdal-docs/blob/master/objects.inv | less


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """
    Used to fix docstrings on-the-fly when generatiing pages with sphinx
    See https://github.com/sphinx-doc/sphinx/issues/10151
    """
    for i in range(len(lines)):
        # Auto convert np.whatever into numpy.whatever in docstrings for sphinx
        lines[i] = lines[i].replace("np.",        "numpy.")
        lines[i] = lines[i].replace("Callable[",  "~typing.Callable[")
        lines[i] = lines[i].replace("Generator[", "~typing.Generator[")
        lines[i] = lines[i].replace("Generic[",   "~typing.Generic[")
        lines[i] = lines[i].replace("Any",        "~typing.Any")
        lines[i] = lines[i].replace("Dict[",      "~typing.Dict[")
        lines[i] = lines[i].replace("Iterator[",  "~typing.Iterator[")
        lines[i] = lines[i].replace("List[",      "~typing.List[")
        lines[i] = lines[i].replace("Literal[",   "~typing.Literal[")
        lines[i] = lines[i].replace("KeysView[",  "~typing.KeysView[")
        lines[i] = lines[i].replace("Optional[",  "~typing.Optional[")
        lines[i] = lines[i].replace("Set[",       "~typing.Set[")
        lines[i] = lines[i].replace("Tuple[",     "~typing.Tuple[")
        lines[i] = lines[i].replace("Type[",      "~typing.Type[")
        lines[i] = lines[i].replace("TypeVar[",   "~typing.TypeVar[")
        lines[i] = lines[i].replace("Union[",     "~typing.Union[")


# Configuration for inheritance_diagram
inheritance_graph_attrs = dict(rankdir="TB")
