"""Stats command for RASSDB - display database statistics.

This module provides the CLI interface for viewing statistics about
the RASSDB vector database.
"""

import json
from typing import Dict, Any
import click
from tabulate import tabulate

from rassdb.vector_store import VectorStore


def format_stats(stats: Dict[str, Any], format: str) -> str:
    """Format statistics for display.
    
    Args:
        stats: Statistics dictionary.
        format: Output format.
        
    Returns:
        Formatted statistics string.
    """
    if format == "json":
        return json.dumps(stats, indent=2)
    
    elif format == "table":
        output = []
        
        # Overall stats
        output.append("Database Statistics")
        output.append("=" * 40)
        output.append(f"Total chunks: {stats['total_chunks']}")
        output.append(f"Unique files: {stats['unique_files']}")
        output.append("")
        
        # By language
        if stats.get('by_language'):
            lang_data = [[lang, count] for lang, count in sorted(stats['by_language'].items())]
            output.append("Chunks by Language:")
            output.append(tabulate(lang_data, headers=["Language", "Count"], tablefmt="grid"))
            output.append("")
        
        # By type
        if stats.get('by_type'):
            type_data = [[chunk_type, count] for chunk_type, count in sorted(stats['by_type'].items())]
            output.append("Chunks by Type:")
            output.append(tabulate(type_data, headers=["Type", "Count"], tablefmt="grid"))
        
        return "\n".join(output)
    
    else:  # simple
        output = []
        output.append(f"Total chunks: {stats['total_chunks']}")
        output.append(f"Unique files: {stats['unique_files']}")
        
        if stats.get('by_language'):
            output.append("\nBy language:")
            for lang, count in sorted(stats['by_language'].items()):
                output.append(f"  {lang}: {count}")
        
        if stats.get('by_type'):
            output.append("\nBy type:")
            for chunk_type, count in sorted(stats['by_type'].items()):
                output.append(f"  {chunk_type}: {count}")
        
        return "\n".join(output)


@click.command(name="rassdb-stats")
@click.option("--db", default="code_rag.db", help="Database file path")
@click.option("--format", "-f",
              type=click.Choice(["table", "json", "simple"]),
              default="table",
              help="Output format")
def main(db: str, format: str) -> None:
    """Display statistics about the RASSDB database.
    
    Shows information about the number of indexed code chunks,
    files, and breakdowns by language and chunk type.
    
    Examples:
    
        # Show statistics in table format
        rassdb-stats
        
        # Use a custom database
        rassdb-stats --db myproject.db
        
        # Output as JSON
        rassdb-stats --format json
    """
    try:
        vector_store = VectorStore(db)
        stats = vector_store.get_statistics()
        vector_store.close()
        
        output = format_stats(stats, format)
        print(output)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


if __name__ == "__main__":
    main()