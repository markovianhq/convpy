# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='convpy',
    version='1.0.0',
    description='Lagged Conversion Prediction.',
    author='Alexander Volkmann',
    packages=['convpy'],
    package_dir={'convpy': 'convpy'},
    package_data={'convpy': ['tests/data/*.csv']},
    url='https://github.com/markovianhq/convpy',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ]

)
