from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rassdb-chat",
    version="0.1.0",
    author="RASSDB Team",
    author_email="team@rassdb.example.com",
    description="A chat bot demonstrating RASSDB MCP integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/rassdb-chat",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "python-multipart>=0.0.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "pydata-sphinx-theme>=0.13.0",
            "sphinx-autodoc-typehints>=1.22.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rassdb-chat-server=rassdb_chat.api.server:main",
        ],
    },
)