#!/usr/bin/env python3
"""Get list of changed files with line numbers from git diff.

Usage:
    git-changed-lines.py                    # All changes in current directory
    git-changed-lines.py file1.py file2.py  # Specific files
    git-changed-lines.py dir1 dir2 file.py  # Multiple directories and files
    git-changed-lines.py -- HEAD^^          # Changes from 2 commits ago
    git-changed-lines.py -- HEAD~3 file.py  # Specific file changes from 3 commits ago
    git-changed-lines.py --cached           # Staged changes
    git-changed-lines.py -- origin/main     # Changes from origin/main
"""

import subprocess
import sys
import re
import os
from typing import List, Tuple, Optional


def parse_diff_output(diff_output: str) -> List[Tuple[str, int]]:
    """Parse git diff output to extract file paths and line numbers.

    Args:
        diff_output: Raw output from git diff command

    Returns:
        List of (file_path, line_number) tuples
    """
    results = []
    current_file = None

    for line in diff_output.split("\n"):
        # Match file header: +++ b/file/path.py
        if line.startswith("+++"):
            match = re.match(r"\+\+\+ b/(.+)", line)
            if match:
                current_file = match.group(1)
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
                    results.append((current_file, start_line + i))

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
        print(f"Error running git diff: {e}", file=sys.stderr)
        print(f"Command: {' '.join(cmd)}", file=sys.stderr)
        if e.stderr:
            print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    args = sys.argv[1:]

    # Parse arguments to separate git options from files/directories
    git_options = []
    files_dirs = []
    parsing_git_options = False

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

    # Parse and print results
    results = parse_diff_output(diff_output)

    for file_path, line_num in results:
        print(f"{file_path}:{line_num}")


if __name__ == "__main__":
    main()
