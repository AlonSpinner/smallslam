from setuptools import setup
from setuptools import find_packages

setup(
    name = "smallslam",
    version = "1.0.0",
    description = "bla bla",
    author = "bla bla",
    url = "https://github.com/AlonSpinner/smallslam",
    packages = find_packages(exclude = ('tests*')),
    )