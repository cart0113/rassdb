"""Database discovery utilities for finding .rassdb files."""

import os
from pathlib import Path
from typing import Optional, List


def find_rassdb_files(start_dir: Path = None) -> List[Path]:
    """Find all .rassdb files in the .rassdb directory.

    Args:
        start_dir: Directory to start searching from. Defaults to current directory.

    Returns:
        List of paths to .rassdb files found.
    """
    if start_dir is None:
        start_dir = Path.cwd()

    rassdb_dir = start_dir / ".rassdb"

    if not rassdb_dir.exists() or not rassdb_dir.is_dir():
        return []

    # Find all .rassdb files in the directory
    rassdb_files = list(rassdb_dir.glob("*.rassdb"))
    return rassdb_files


def discover_database(db_path: Optional[str] = None, start_dir: Path = None) -> str:
    """Discover the appropriate database file to use.

    Args:
        db_path: Explicitly provided database path. If provided, this is used directly.
        start_dir: Directory to start searching from. Defaults to current directory.

    Returns:
        Path to the database file to use.

    Raises:
        FileNotFoundError: If no database is found.
        ValueError: If multiple databases are found and no specific one is provided.
    """
    # If explicit path is provided, use it
    if db_path:
        return db_path

    # Look for .rassdb files in the .rassdb directory
    rassdb_files = find_rassdb_files(start_dir)

    if len(rassdb_files) == 0:
        raise FileNotFoundError(
            "No database found in .rassdb directory. Please specify a database "
            "using --db or ensure you're in a directory with a .rassdb folder."
        )

    if len(rassdb_files) == 1:
        # Exactly one database found - use it
        return str(rassdb_files[0])

    # Multiple databases found
    db_names = [f.name for f in rassdb_files]
    raise ValueError(
        f"Multiple databases found in .rassdb directory: {', '.join(db_names)}. "
        f"Please specify which database to use with --db."
    )
