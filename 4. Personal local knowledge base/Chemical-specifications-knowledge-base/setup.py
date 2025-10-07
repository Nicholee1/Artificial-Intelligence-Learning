#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Chemical Specifications Knowledge Base
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chemical-specifications-knowledge-base",
    version="1.0.0",
    author="Chemical KB Team",
    author_email="contact@chemical-kb.com",
    description="A comprehensive knowledge base system for chemical engineering documents with AI-powered search and question answering capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/chemical-kb/chemical-specifications-knowledge-base",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chemical-kb-chat=chemical_kb.ai.chat:main",
            "chemical-kb-api=chemical_kb.api.server:main",
            "chemical-kb-config=chemical_kb.utils.config:main",
            "chemical-kb-pipeline=chemical_kb.core.pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "chemical_kb": ["config/*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/chemical-kb/chemical-specifications-knowledge-base/issues",
        "Source": "https://github.com/chemical-kb/chemical-specifications-knowledge-base",
        "Documentation": "https://chemical-kb.readthedocs.io/",
    },
)
