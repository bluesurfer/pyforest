#!/usr/bin/env python

from distutils.core import setup

setup(
    name='pyForest',
    version='1.0',
    author=['Andrea Casini'],
    email='acasini@unive.dsi.it',
    packages=['pyforest'],
    license='LICENSE',
    description='Random Forests for face detection',
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'PIL',
                      'scikit-learn'],
    long_description=open('README.rst').read()
)
