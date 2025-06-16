"""Index command for RASSDB - build code embeddings database.

This module provides the CLI interface for indexing codebases into
the RASSDB vector database.
"""

import logging
from pathlib import Path
import click

from rassdb.indexer import CodebaseIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


@click.command(name="rassdb-index")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--db", default="code_rag.db", help="Database file path")
@click.option("--no-gitignore", is_flag=True, help="Do not use .gitignore")
@click.option("--clear", is_flag=True, help="Clear existing database before indexing")
@click.option("--model", default="nomic-ai/nomic-embed-text-v1.5", help="Embedding model name")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def main(
    directory: str,
    db: str,
    no_gitignore: bool,
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
        
        # Create indexer
        indexer = CodebaseIndexer(
            db_path=db,
            model_name=model,
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