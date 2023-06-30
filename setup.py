#!/usr/bin/env python

import os

from setuptools import find_packages
from setuptools import setup


def get_all_subdirs(top, *dirs):
    dir = os.path.join(*dirs)
    top = os.path.abspath(top)
    data_dir = []
    for dir, sub_dirs, files in os.walk(os.path.join(top, dir)):
        data_dir.append(dir[len(top) + 1:])
    data_dir = list(filter(lambda x: not os.path.basename(x).startswith('_'), data_dir))
    data_dir = list(map(lambda x: os.path.join(x, '*'), data_dir))
    return data_dir


__minimum_jax_version__ = '0.2.9'

setup_requires = [
    'jax>=' + __minimum_jax_version__,
    'jaxns>=2.2.1'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='jaxns_cosmology',
      version='0.0.1',
      description='JAXNS Cosmology',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Joshuaalbert/jaxns-cosmology",
      author='Joshua G. Albert',
      author_email='josh.albert@touchintel.io',
      setup_requires=setup_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.10'
      )
