#!/usr/bin/env python

from distutils.core import setup

setup(
    name='toy_data',
    version='0.2dev',
    description='Toy data for machine learning projects',
    author='Chen Yu',
    author_email='chenyu.nus@gmail.com',
    url='https://github.com/cwhy/toy_data',
    packages=['toy_data', 'toy_data.test'],
    install_requires=[
        "numpy",
        "bokeh",
    ],
)
