"""Setup script for RASSDB."""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rassdb",
    version="0.1.0",
    author="AJ Carter",
    author_email="ajcarter@example.com",
    description="Lightweight code RAG/semantic search database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajcarter/rassdb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "click>=8.0",
        "sentence-transformers>=2.0",
        "sqlite-vec>=0.1.1",
        "tree-sitter>=0.20",
        "tree-sitter-python>=0.20",
        "tree-sitter-javascript>=0.20",
        "tree-sitter-typescript>=0.20",
        "tree-sitter-java>=0.20",
        "tree-sitter-cpp>=0.20",
        "tree-sitter-c>=0.20",
        "tree-sitter-rust>=0.20",
        "tree-sitter-go>=0.20",
        "gitignore-parser>=0.1",
        "tqdm>=4.0",
        "numpy>=1.20",
        "tabulate>=0.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1",
            "mypy>=1.0",
            "sphinx>=6.0",
            "pydata-sphinx-theme>=0.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "rassdb-search=rassdb.cli.search:main",
            "rassdb-index=rassdb.cli.index:main",
            "rassdb-stats=rassdb.cli.stats:main",
        ],
    },
)