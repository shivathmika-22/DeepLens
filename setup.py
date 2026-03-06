"""
Setup script for DeepLens package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirement.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deeplens",
    version="1.0.0",
    author="DeepLens Team",
    author_email="team@deeplens.ai",
    description="A comprehensive news intelligence platform for multi-source data aggregation and analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deeplens",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "api": [
            "fastapi[all]>=0.85.0",
            "python-jose>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deeplens=run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "deeplens": ["config/*.py", "utils/*.py"],
    },
)
