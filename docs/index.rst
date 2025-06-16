RASSDB Documentation
====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   cli
   mcp

Introduction
------------

RASSDB (Retrieval-Augmented Search and Semantic Database) is a lightweight, efficient database designed for code retrieval-augmented generation (RAG) and semantic search.

Features
--------

* **Dual Search Modes**: Both semantic (embedding-based) and literal (grep-like) search
* **Tree-sitter Integration**: Intelligent code parsing into semantic chunks
* **Fast Vector Search**: Uses sqlite-vec for efficient similarity search
* **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C/C++, Rust, Go, and more
* **MCP Ready**: Designed to work as an MCP (Model Context Protocol) server
* **Simple CLI**: Easy-to-use command-line tools for indexing and searching

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`