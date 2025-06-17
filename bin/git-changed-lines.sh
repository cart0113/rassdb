#!/bin/bash

# Script to get list of changed files with line numbers from git diff
# Usage: 
#   git-changed-lines.sh                    # Check all changes in current directory
#   git-changed-lines.sh file1.py file2.py  # Check specific files
#   git-changed-lines.sh dir1 dir2 file.py  # Check multiple directories and files

# Function to process git diff output
process_diff() {
    grep -E "^@@|^\+\+\+" | awk '
/^\+\+\+/ {
    file = $2
    sub(/^b\//, "", file)
}
/^@@/ {
    # Extract line numbers from @@ format
    split($3, range, ",")
    sub(/^\+/, "", range[1])
    start = range[1]
    count = range[2] ? range[2] : 1
    for (i = 0; i < count; i++) {
        if (file != "/dev/null") {
            print file ":" (start + i)
        }
    }
}'
}

# If no arguments, check all changes in current directory
if [[ $# -eq 0 ]]; then
    git diff --unified=0 | process_diff
else
    # Process each argument
    for arg in "$@"; do
        if [[ -d "$arg" ]]; then
            # It's a directory - run git diff for that directory
            git diff --unified=0 "$arg" | process_diff
        else
            # It's a file - run git diff for that file
            git diff --unified=0 "$arg" | process_diff
        fi
    done
fi