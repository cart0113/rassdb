Configuration
=============

RASSDB uses TOML files for configuration, allowing fine-grained control over which files are indexed and which embedding model is used.

Configuration File Precedence
-----------------------------

RASSDB looks for configuration files in the following order:

1. **Project-specific config**: ``.rassdb-config.toml`` in the project root directory
2. **Global config**: ``~/.rassdb-config.toml`` in your home directory

**Important**: If a project-specific config exists, it completely overrides the global config. There is no merging of configurations.

Configuration Options
---------------------

embedding-model
~~~~~~~~~~~~~~~

Specifies which embedding model to use for indexing:

.. code-block:: toml

    [embedding-model]
    name = "nomic-ai/nomic-embed-text-v1.5"

Popular embedding models:

- ``nomic-ai/nomic-embed-text-v1.5`` - Balanced performance (768 dimensions)
- ``nomic-ai/nomic-embed-code-gguf`` - GGUF code-specific model
- ``sentence-transformers/all-MiniLM-L6-v2`` - Small, fast general-purpose
- ``Qodo/Qodo-Embed-1-1.5B`` - Structured metadata-based retrieval
- ``voyage-code-2`` - Voyage AI code model (requires API key)
- ``text-embedding-3-small`` - OpenAI model (requires API key)

include-extensions
~~~~~~~~~~~~~~~~~~

Specifies which file extensions to index using **regex patterns**:

.. code-block:: toml

    [include-extensions]
    python = ["\.py$", "\.pyi$"]
    javascript = ["\.js$", "\.jsx$", "\.mjs$", "\.cjs$"]
    typescript = ["\.ts$", "\.tsx$"]
    go = ["\.go$"]
    rust = ["\.rs$"]
    markdown = ["\.md$", "\.markdown$"]
    restructuredtext = ["\.rst$"]

**Important**: These are regex patterns, not simple extensions. Use ``\.ext$`` to match files ending with ``.ext``.

If no ``include-extensions`` is specified, **all files** in the directory will be indexed.

include-paths
~~~~~~~~~~~~~

Specifies glob patterns for paths to include:

.. code-block:: toml

    include-paths = ["src/**/*", "lib/**/*", "test/**/*"]

exclude-extensions
~~~~~~~~~~~~~~~~~~

Specifies file extensions to exclude (regex patterns):

.. code-block:: toml

    [exclude-extensions]
    binary = ["\.pyc$", "\.pyo$", "\.so$", "\.dll$", "\.dylib$"]
    data = ["\.db$", "\.sqlite$", "\.json$", "\.xml$", "\.csv$"]

exclude-paths
~~~~~~~~~~~~~

Specifies glob patterns for paths to exclude:

.. code-block:: toml

    exclude-paths = ["**/node_modules/**", "**/__pycache__/**", "**/build/**"]

Include/Exclude Priority
------------------------

When both include and exclude rules are present:

1. **Include always wins**: If a file matches any include rule, it will be indexed
2. Files must match include rules AND not match exclude rules
3. If no config is present, all files are indexed (no filtering)

Example Configurations
----------------------

Minimal Python Project
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

    [include-extensions]
    python = ["\.py$"]

Web Development Project
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

    [embedding-model]
    name = "nomic-ai/nomic-embed-text-v1.5"

    [include-extensions]
    python = ["\.py$"]
    javascript = ["\.js$", "\.jsx$", "\.mjs$"]
    css = ["\.css$", "\.scss$"]
    html = ["\.html$", "\.htm$"]
    
    exclude-paths = [
        "**/node_modules/**",
        "**/build/**",
        "**/dist/**"
    ]

Documentation-Heavy Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

    [include-extensions]
    python = ["\.py$"]
    markdown = ["\.md$", "\.markdown$"]
    restructuredtext = ["\.rst$"]
    text = ["\.txt$"]
    yaml = ["\.yml$", "\.yaml$"]

No Configuration
~~~~~~~~~~~~~~~~

If no configuration file is present, RASSDB will:

- Use the default embedding model
- Index **all files** in the directory tree
- Apply no filtering based on extensions or paths

This makes it easy to index entire codebases without any setup.