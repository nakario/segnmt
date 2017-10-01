#!/usr/bin/env python

import sys

from setuptools import find_packages
from setuptools import setup


if sys.version_info < (3, 6):
    sys.exit('This program needs Python 3.6 or later version.')


setup(
    name='segnmt',
    version='0.1.0',
    packages=find_packages(),
    requires=['chainer>=2.1.0', 'matplotlib', 'progressbar2']
)
