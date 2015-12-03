from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,4):
    print("nnets requires Python 3.4 or later", file=sys.stderr)
    sys.exit(1)

with open("README.md") as f:
    longdesc = f.read()

setup (name = 'nnets',
       version = '0.1',
       description = "Neural Network representation.",
       author = 'Zahari Kassabov',
       author_email = 'kassabov@to.infn.it',
       url = 'https://github.com/Zaharid/nnets',
       long_description = longdesc,
       package_dir = {'': 'src'},
       packages = find_packages('src'),
       package_data = {
            '':[]
       },
       zip_safe = False,
       classifiers=[
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            ],
       )
