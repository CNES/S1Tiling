#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================================
#   Program:   S1Processor
#
#   Copyright 2017-2020 (c) CESBIO. All rights reserved.
#
#   This file is part of S1Tiling project
#       https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# =========================================================================
#
# Authors: Thierry KOLECK (CNES)
#          Luc HERMITTE (CS Group)
#
# =========================================================================

import os
from setuptools import setup, find_packages

# Import the library to make sure there is no side effect
import s1tiling

BASEDIR = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

metadata = {}
with open(os.path.join(BASEDIR, "s1tiling", "__meta__.py"), "r") as f:
    exec(f.read(), metadata)

with open(os.path.join(BASEDIR, "README.md"), "r") as f:
    readme = f.read()

setup(
    name             = metadata["__title__"],
    version          = metadata["__version__"],
    description      = metadata["__description__"],
    long_description = readme,
    author           = metadata["__author__"],
    author_email     = metadata["__author_email__"],
    url              = metadata["__url__"],
    license          = metadata["__license__"],
    keywords         = "Sentinel-1, Sentinel-2, orthorectification",

    # Liste les packages à insérer dans la distribution
    # plutôt que de le faire à la main, on utilise la foncton
    # find_packages() de setuptools qui va cherche tous les packages
    # python recursivement dans le dossier courant.
    # C'est pour cette raison que l'on a tout mis dans un seul dossier:
    # on peut ainsi utiliser cette fonction facilement
    packages=find_packages(exclude=("*.tests", "*.tests.*", "tests.*", "tests")),
    package_data={"": ["LICENSE", "NOTICE"]},
    include_package_data=True, # Take MANIFEST.in into account

    python_requires='>=3.3, <4',
    install_requires=[
        "dask[distributed]",
        "eodag",
        "numpy",
        "ogr",
        "osgeo",
        "pickle",
        "rasterio",
        # Any way to require OTB ?
        ],
    extras_require={
        "dev": [
            # "nose",
            # "tox",
            # "faker",
            # 'mock; python_version < "3.5" ',
            # "coverage",
            # "moto==1.3.6",
            # "twine",
            "wheel",
            "flake8",
            "pre-commit",
            ],
        "docs": ["sphinx == 1.8.0", "nbsphinx == 0.3.5", "nbsphinx-link == 1.1.1"],
        },

    # https://pypi.python.org/pypi?%3Aaction=list_classifiers.
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: GIS",
        ],

    project_urls={
            "Bug Tracker": "https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling/-/issues",
            "Documentation": "https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling",
            "Source Code": "https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling",
            },

    scripts = ['s1tiling/S1Processor.py'],

    # C'est un système de plugin, mais on s'en sert presque exclusivement
    # Pour créer des commandes, comme "django-admin".
    # Par exemple, si on veut créer la fabuleuse commande "proclame-sm", on
    # va faire pointer ce nom vers la fonction proclamer(). La commande sera
    # créé automatiquement.
    # La syntaxe est "nom-de-commande-a-creer = package.module:fonction".
    entry_points = {
        'console_scripts': [
            'S1Tiling.py = s1tiling.core:proclamer',
        ],
    },
)
