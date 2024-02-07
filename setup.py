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


# Install requires:
# jaxns==2.4.8
# jax
# jaxlib
# etils
# bilby
# matplotlib
# numpy
# scipy
# pytest
# tensorflow_probability
# arviz
# nessai
# pyyaml
# h5py
# pydantic
# nautilus-sampler
# ultranest
# dynesty
# pymultinest
# git+https://github.com/PolyChord/PolyChordLite@master

install_requires = [
    "jaxns==2.4.8",
    "jax",
    "jaxlib",
    "etils",
    "bilby",
    "matplotlib",
    "numpy",
    "scipy",
    "tensorflow_probability",
    "arviz",
    "nessai",
    "pyyaml",
    "h5py",
    "pydantic",
    "nautilus-sampler",
    "ultranest",
    "dynesty",
    "pymultinest",
    "pypolychord"
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
      install_requires=install_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_data={
          'jaxns_cosmology': sum([get_all_subdirs('jaxns_cosmology', 'models', 'ml_functions')], [])
      },
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.10'
      )
