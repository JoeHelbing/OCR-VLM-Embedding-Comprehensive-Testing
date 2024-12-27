#!/bin/bash

# Check if a repository URL is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <github_repo_url>"
    exit 1
fi

# GitHub repository URL
repo_url="$1"

# Extract repository name from the URL (remove .git if present)
repo_name=$(basename -s .git "$repo_url")

# Output file name
output_file="${repo_name}_context.md"

# Remove the output file if it already exists
rm -f "$output_file"

# Function to process each file
process_file() {
    local file="$1"
    echo "Processing file: $file"
    echo "Path: $file" >> "$output_file"
    echo "" >> "$output_file"

    # Determine the file extension
    extension="${file##*.}"

    case "$extension" in
        ipynb)
            # Use jq to parse the notebook and extract only code/markdown cells
            echo "Extracting Python code and Markdown from $file"
            jq -r '
                .cells[] |
                if .cell_type == "markdown" then
                  "```markdown\n" + (.source | join("")) + "\n```"
                elif .cell_type == "code" then
                  "```python\n" + (.source | join("")) + "\n```"
                else
                  empty
                end
            ' "$file" >> "$output_file"
            ;;
        toml)
            language="toml"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        md)
            language=""
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        R|Rmd)
            language="r"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        rs)
            language="rust"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        cpp|cxx|cc)
            language="cpp"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        h|hpp)
            language="cpp"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        py)
            language="python"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
        *)
            # Default to plain text if no case is matched
            language="plaintext"
            echo "\`\`\`$language" >> "$output_file"
            cat "$file" >> "$output_file"
            echo "\`\`\`" >> "$output_file"
            ;;
    esac

    echo "" >> "$output_file"
    echo "-----------" >> "$output_file"
    echo "" >> "$output_file"
}

# Check if required tools are installed
# Note: Replace 'fd' with 'fdfind' here
for tool in git fdfind jq; do
    if ! command -v "$tool" &> /dev/null; then
        echo "Error: $tool is not installed. Please install $tool and try again."
        exit 1
    fi
done

# Create a temporary directory for cloning
temp_dir=$(mktemp -d)

# Clone the repository
echo "Cloning repository..."
git clone "$repo_url" "$temp_dir"

# Change to the cloned repository directory
cd "$temp_dir" || exit 1

# Use 'fdfind' instead of 'fd'
# Remove '-H' if your version of fdfind does not recognize it, or keep if it supports hidden files
fdfind -H -t f \
   -e toml -e rs -e qml -e cpp -e h -e md -e R -e Rmd -e ipynb -e py -e cxx -e cc -e hpp |
    sort -n -t'/' -k'1' |
    while read -r file; do
        process_file "$file"
    done

# Move the output file to the original directory
mv "$output_file" "$OLDPWD"

# Change back to the original directory
cd "$OLDPWD" || exit 1

# Clean up: remove the temporary directory
rm -rf "$temp_dir"

echo "Repository contents have been processed and combined into $output_file"
