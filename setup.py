import os

from setuptools import find_packages, setup


setup(
    name="ml-pytorch",
    version='1.0',
    packages=find_packages(exclude=('tests')),
    install_requires=open(os.path.abspath(os.path.join(os.path.dirname(__file__), "requirements.txt")), "r").read().strip(),
    author="Thomas Beucher",
    author_email="beucherlamacque.thomas@gmail.com",
    description="Personal Pytorch Toolkit"
)