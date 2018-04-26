# this script was pulled from the following link
#  https://jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/
# and I'm not totally sure it is set up to work that well with the package 
# in its current form. We'll see!

from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

import ccbid

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

#long_description = read('README.txt')

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='ccb-id',
    version=ccbid.__version__,
    url='https://github.com/stanford-ccb/ccb-id/',
    license='MIT',
    author='Christopher Anderson',
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    author_email='cbanders@stanford.edu',
    description='Species classification approach using imaging spectroscopy',
    long_description=long_description,
    packages=['ccbid'],
    include_package_data=True,
    platforms='any',
    scripts=['bin/train.py']
)
