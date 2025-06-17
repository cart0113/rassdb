"""Index command for RASSDB - build code embeddings database.

This module provides the CLI interface for indexing codebases into
the RASSDB vector database.
"""

import logging
from pathlib import Path
import click

from rassdb.indexer import CodebaseIndexer


def get_default_db_path(directory: Path, model_name: str) -> Path:
    """Generate the default database path based on directory and model name.

    Creates a path like: {directory}/.rassdb/{folder_name}-{model_name}.rassdb
    """
    dir_path = Path(directory).resolve()
    folder_name = dir_path.name

    # Extract just the model name from the full model path
    # e.g., "nomic-ai/nomic-embed-text-v1.5" -> "nomic-embed-text-v1.5"
    model_short_name = model_name.split("/")[-1]

    # Create the .rassdb directory path
    rassdb_dir = dir_path / ".rassdb"

    # Generate the database filename
    db_filename = f"{folder_name}-{model_short_name}.rassdb"

    return rassdb_dir / db_filename


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@click.command(name="rassdb-index")
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--db",
    default=None,
    help="Database file path (default: {dir}/.rassdb/{dir}-{model}.rassdb)",
)
@click.option("--no-gitignore", is_flag=True, help="Do not use .gitignore")
@click.option(
    "--no-rassdb-config",
    is_flag=True,
    help="Do not use .rassdb-config.toml",
)
@click.option("--clear", is_flag=True, help="Clear existing database before indexing")
@click.option(
    "--model", default="nomic-ai/CodeRankEmbed", help="Embedding model name"
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def main(
    directory: str,
    db: str | None,
    no_gitignore: bool,
    no_rassdb_config: bool,
    clear: bool,
    model: str,
    quiet: bool,
) -> None:
    """Index a codebase directory into the RASSDB database.

    This command scans a directory for code files, parses them into semantic
    chunks using Tree-sitter, generates embeddings, and stores them in a
    SQLite database with vector search capabilities.

    Examples:

        # Index a project
        rassdb-index ~/my-project

        # Clear existing data and re-index
        rassdb-index ~/my-project --clear

        # Use a custom database file
        rassdb-index ~/my-project --db myproject.db

        # Ignore .gitignore rules
        rassdb-index ~/my-project --no-gitignore
    """
    try:
        # Set logging level based on quiet flag
        if quiet:
            logging.getLogger().setLevel(logging.WARNING)

        # Use default path if no db path provided
        if db is None:
            db_path = get_default_db_path(directory, model)
        else:
            db_path = db

        # Create indexer
        indexer = CodebaseIndexer(
            db_path=str(db_path),
            model_name=model,
            use_rassdb_config=not no_rassdb_config,
        )

        # Index the directory
        indexer.index_directory(
            directory,
            use_gitignore=not no_gitignore,
            show_progress=not quiet,
            clear_existing=clear,
        )

        # Clean up
        indexer.close()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


if __name__ == "__main__":
    main()
