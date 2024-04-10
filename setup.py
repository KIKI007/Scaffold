#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import re
import io

import argparse
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

def read(*names, **kwargs):
    return io.open(
        os.path.join(here, *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()


about = {}
exec(read('python', 'scaffold', '__version__.py'), about)

requirements = [
    "gurobipy==10.0.2",
    "jax==0.4.21",
    "polyscope==2.2.1",
    "libigl==2.4.1",
    "compas_eve",
    "termcolor",
    "jaxlib==0.4.21",
    "distance3d",
    "open3d"
]

ext_modules = []

setup(
    name=about['__title__'],
    version=about['__version__'],
    license=about['__license__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    long_description='',
    packages=find_packages('python'),
    package_dir={'': 'python'},
    # zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.10',
    ],
    keywords=[''],
    install_requires=requirements,
)
