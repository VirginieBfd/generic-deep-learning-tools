"""
Generic Deep Learning Tools For Computer Vision Applications
"""


import re
from codecs import open
from os.path import abspath, dirname, join
from setuptools import find_packages, setup
this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'tools', 'generic_dl_tools', '__init__.py'),
          encoding='utf-8') as version_file:
    version_number = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                               version_file.read(), re.MULTILINE).group(1)

if not version_number:
    raise RuntimeError('Cannot find version information')

setup(
    name='generic-deep-learning-tools-for-cv',
    version=version_number,
    url='https://github.com/VirginieBfd/generic-deep-learning-tools-for-cv',
    author='VirginieBfd',
    classifiers=[
        'Topic :: Utilities',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['deep learning', 'metrics'],
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy',
        'opencv-python',
        'keras'
    ],
)