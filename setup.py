import subprocess
import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='alpha_continuum',
    version='0.0.1',
    description='The code for spectral continuum normalization using alpha shape method.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/MingjieJian/alpha_continuum',
    author='Mingjie Jian',
    author_email='ssaajianmingjie@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Framework :: IPython",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    python_requires=">=3.5",
    packages=setuptools.find_packages(exclude=['files']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
    ])