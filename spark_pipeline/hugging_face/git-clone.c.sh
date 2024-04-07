#!/bin/bash

# Get the directory where the script resides
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $script_dir"
ls $script_dir

# File containing repository URLs (one per line)
url_file="$script_dir/model_repo_urls.txt"  

# Check if the file exists
if [ ! -f "$url_file" ]; then
    echo "File '$url_file' not found. Exiting..."
    exit 1
fi

echo "Reading files..."

# Read each line from the file and clone the repository
while IFS= read -r url || [[ -n "$url" ]]; do
    echo "Cloning repository from: $url"
    git clone "$url"
    # Check if the clone was successful
    if [ $? -eq 0 ]; then
        echo "Repository cloned successfully."
    else
        echo "Failed to clone repository."
    fi
done < "$url_file"
