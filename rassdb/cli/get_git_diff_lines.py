"""Get list of changed files with line numbers from git diff.

This module provides the CLI interface for extracting changed lines from git diffs,
optionally with context from the RASSDB database.
"""

import subprocess
import sys
import re
import os
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
import click
import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_rassdb_in_folder(folder: Path) -> Optional[Path]:
    """Find a single .rassdb file in the .rassdb folder.

    Args:
        folder: Directory to search in

    Returns:
        Path to the database file if exactly one found, None otherwise
    """
    rassdb_dir = folder / ".rassdb"
    if not rassdb_dir.exists():
        return None

    db_files = list(rassdb_dir.glob("*.rassdb"))
    if len(db_files) == 1:
        return db_files[0]
    elif len(db_files) > 1:
        raise click.ClickException(
            f"Multiple .rassdb files found in {rassdb_dir}. Please specify --db-path."
        )
    return None


def parse_diff_output(diff_output: str) -> Dict[str, Set[int]]:
    """Parse git diff output to extract file paths and line numbers.

    Args:
        diff_output: Raw output from git diff command

    Returns:
        Dict mapping file paths to sets of changed line numbers
    """
    results = {}
    current_file = None

    for line in diff_output.split("\n"):
        # Match file header: +++ b/file/path.py
        if line.startswith("+++"):
            match = re.match(r"\+\+\+ b/(.+)", line)
            if match:
                current_file = match.group(1)
                if current_file not in results:
                    results[current_file] = set()
            else:
                # Handle /dev/null case
                current_file = None

        # Match hunk header: @@ -old +new,count @@
        elif line.startswith("@@") and current_file:
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                start_line = int(match.group(1))
                count = int(match.group(2)) if match.group(2) else 1

                # Add all lines in this hunk
                for i in range(count):
                    results[current_file].add(start_line + i)

    return results


def run_git_diff(git_args: List[str], paths: List[str]) -> str:
    """Run git diff command and return output.

    Args:
        git_args: Arguments to pass to git diff (e.g., ['HEAD~1', '--cached'])
        paths: File or directory paths to check

    Returns:
        Raw git diff output
    """
    cmd = ["git", "diff", "--unified=0"] + git_args + paths

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running git diff: {e}")
        logger.error(f"Command: {' '.join(cmd)}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        sys.exit(1)


def query_code_chunks_for_lines(
    db_path: Path, file_path: str, line_numbers: Set[int]
) -> List[Dict[str, Any]]:
    """Query the database for code chunks that contain the given line numbers.

    Args:
        db_path: Path to the RASSDB database
        file_path: Path to the file
        line_numbers: Set of line numbers to find chunks for

    Returns:
        List of matching code chunks with their metadata
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Convert line numbers to a sorted list for easier processing
    sorted_lines = sorted(line_numbers)

    # Query for all chunks that might contain these lines
    # We need chunks where any of our line numbers fall within [part_start_line, part_end_line]
    placeholders = ",".join(["?"] * len(sorted_lines))
    query = f"""
        SELECT DISTINCT 
            id, file_path, content, language, 
            start_line, end_line, chunk_type, metadata,
            part_start_line, part_end_line
        FROM code_chunks
        WHERE file_path = ?
        AND (
            {' OR '.join(['(part_start_line <= ? AND part_end_line >= ?)'] * len(sorted_lines))}
        )
        ORDER BY part_start_line
    """

    # Build parameters: file_path, then pairs of (line, line) for each line number
    params = [file_path]
    for line in sorted_lines:
        params.extend([line, line])

    cursor.execute(query, params)

    chunks = []
    for row in cursor.fetchall():
        chunk = {
            "id": row[0],
            "file_path": row[1],
            "content": row[2],
            "language": row[3],
            "start_line": row[4],
            "end_line": row[5],
            "chunk_type": row[6],
            "metadata": json.loads(row[7]) if row[7] else {},
            "part_start_line": row[8],
            "part_end_line": row[9],
        }
        chunks.append(chunk)

    conn.close()
    return chunks


def get_database_root_path(db_path: Path) -> Optional[str]:
    """Get the root path stored in the database metadata.

    Args:
        db_path: Path to the RASSDB database

    Returns:
        The stored root path if found, None otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT value FROM database_metadata WHERE key = 'root_path'")
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        # Table might not exist in older databases
        return None
    finally:
        conn.close()


def format_output_with_context(
    file_changes: Dict[str, Set[int]], db_path: Optional[Path]
) -> None:
    """Format and print the results with context from the database.

    Args:
        file_changes: Dict mapping file paths to sets of changed line numbers
        db_path: Path to the RASSDB database (if available)
    """
    if not db_path:
        # Simple output without context
        for file_path, line_numbers in file_changes.items():
            for line_num in sorted(line_numbers):
                print(f"{file_path}:{line_num}")
        return

    # Output with context from database
    output_data = []
    match_index = 0

    # Get the stored root path from the database
    db_root_path = get_database_root_path(db_path)

    # Get current working directory
    cwd = Path.cwd().resolve()

    for file_path, line_numbers in file_changes.items():
        # Try different path transformations to find the file in the database
        chunks = []
        query_path = file_path

        # First, try the original path
        chunks = query_code_chunks_for_lines(db_path, query_path, line_numbers)

        # If no results and we have a stored root path, try transforming the path
        if not chunks and db_root_path:
            db_root = Path(db_root_path).resolve()

            # Convert git diff path to absolute path
            abs_file_path = (cwd / file_path).resolve()

            # If the file is under the database root, get the relative path
            try:
                relative_path = abs_file_path.relative_to(db_root)
                query_path = str(relative_path)
                chunks = query_code_chunks_for_lines(db_path, query_path, line_numbers)
            except ValueError:
                # File is not under the database root
                pass

        # Fallback: If still no results, try removing common prefixes
        if not chunks:
            # Check if the file path has a directory component that matches the database location
            path_parts = file_path.split("/")
            if len(path_parts) > 1:
                # Try without the first directory component
                query_path = "/".join(path_parts[1:])
                chunks = query_code_chunks_for_lines(db_path, query_path, line_numbers)

        # Group chunks and find which diff lines they contain
        for chunk in chunks:
            # Find which diff lines are in this chunk
            chunk_diff_lines = [
                line
                for line in line_numbers
                if chunk["part_start_line"] <= line <= chunk["part_end_line"]
            ]

            if chunk_diff_lines:  # Only include if it contains diff lines
                entry = {
                    "path": file_path,
                    "partial_lines": [chunk["part_start_line"], chunk["part_end_line"]],
                    "full_lines": [chunk["start_line"], chunk["end_line"]],
                    "diff_lines": sorted(chunk_diff_lines),
                    "metadata": chunk["metadata"],
                }

                # Add chunk type if available
                if chunk["chunk_type"]:
                    entry["metadata"]["chunk_type"] = chunk["chunk_type"]

                output_data.append({f"match-{match_index}": entry})
                match_index += 1

    print(json.dumps(output_data, indent=2))


@click.command(name="rassdb-get-git-diff-lines")
@click.argument("paths", nargs=-1, type=click.Path())
@click.option(
    "--with-rassdb-context", is_flag=True, help="Include context from RASSDB database"
)
@click.option(
    "--rassdb-path", type=click.Path(exists=True), help="Path to RASSDB database file"
)
@click.option("--cached", is_flag=True, help="Show staged changes")
@click.pass_context
def main(ctx, paths, with_rassdb_context, rassdb_path, cached):
    """Get list of changed files with line numbers from git diff.

    Examples:
        rassdb-get-git-diff-lines                    # All changes in current directory
        rassdb-get-git-diff-lines file1.py file2.py  # Specific files
        rassdb-get-git-diff-lines dir1 dir2 file.py  # Multiple directories and files
        rassdb-get-git-diff-lines -- HEAD^^          # Changes from 2 commits ago
        rassdb-get-git-diff-lines -- HEAD~3 file.py  # Specific file changes from 3 commits ago
        rassdb-get-git-diff-lines --cached           # Staged changes
        rassdb-get-git-diff-lines -- origin/main     # Changes from origin/main
        rassdb-get-git-diff-lines --with-rassdb-context     # Include RASSDB context
        rassdb-get-git-diff-lines --rassdb-path db.rassdb --with-rassdb-context  # Specify database path
    """
    # Handle database path logic if --with-rassdb-context is used
    if with_rassdb_context:
        if rassdb_path:
            # Validate that the provided path is a .rassdb file
            db_path = Path(rassdb_path)
            if not str(db_path).endswith(".rassdb"):
                raise click.ClickException(
                    f"Database path must be a .rassdb file: {db_path}"
                )
        else:
            # Try to find database in current directory
            current_dir = Path.cwd()
            db_path = find_rassdb_in_folder(current_dir)
            if not db_path:
                raise click.ClickException(
                    "No .rassdb file found in .rassdb folder. Please specify --rassdb-path."
                )
            logger.info(f"Using database: {db_path}")
    else:
        db_path = None

    # Parse arguments to separate git options from files/directories
    git_options = []
    files_dirs = []
    parsing_git_options = False

    # Convert paths tuple to list for processing
    args = list(paths)

    # Add --cached if specified
    if cached:
        git_options.append("--cached")

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--":
            # Everything after -- is treated as git options until we hit a file/dir
            parsing_git_options = True
            i += 1
            continue

        if parsing_git_options or arg.startswith("-"):
            # This is a git option
            git_options.append(arg)
            # Check if this is a git revision specifier
            if not arg.startswith("-") and i + 1 < len(args):
                # If the next arg exists and is a file/dir, stop parsing git options
                next_arg = args[i + 1]
                if os.path.exists(next_arg):
                    parsing_git_options = False
        else:
            # This is a file or directory
            files_dirs.append(arg)
            parsing_git_options = False

        i += 1

    # Run git diff
    diff_output = run_git_diff(git_options, files_dirs)

    # Parse results
    file_changes = parse_diff_output(diff_output)

    # Format and print output
    format_output_with_context(file_changes, db_path)


if __name__ == "__main__":
    main()
