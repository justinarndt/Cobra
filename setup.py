"""
This is the setup script for the Cobra project.
"""

from setuptools import setup, find_packages

setup(
    name="cobra",
    version="0.1.0",
    packages=find_packages(),
    description="A unified architecture for accelerated Python.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)