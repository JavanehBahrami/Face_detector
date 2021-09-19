import os

from setuptools import setup


def read_requirements(path='./requirements.txt'):
    with open(path) as file:
        install_requires = file.readlines()

    return install_requires


setup(
    name="detectnet_v2",
    version="0.0.2",
    author="Javaneh Bahrami",
    author_email="bahramisaeede@gmail.com",
    description=(
        "A demonstration of how to use peoplenet model."
    ),
    packages=[
        'detectnet_v2'
    ],
    install_requires=read_requirements()
)
