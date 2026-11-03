#!/usr/bin/env bash

set -euo pipefail

# Check if at least target folders and the new value were provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <new_label_path> <folder1> [folder2 ...]"
  echo "Example: $0 \"/new/path/to/labels\" folderA folderB"
  exit 1
fi

NEW_VALUE="$1"
shift
FOLDERS=("$@")

# Base directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if yq is installed (supports Mike Farah's yq v4+)
HAS_YQ=false
if command -v yq >/dev/null 2>&1; then
  HAS_YQ=true
fi

for folder in "${FOLDERS[@]}"; do
  TARGET_PATH="${SCRIPT_DIR}/${folder}"

  if [ ! -d "$TARGET_PATH" ]; then
    echo "Warning: Directory '${folder}' not found at '${TARGET_PATH}'. Skipping..."
    continue
  fi

  echo "==> Processing folder: ${folder}"

  # Find all .yaml and .yml files (handles spaces in file paths safely)
  while IFS= read -r -d '' file; do
    if [ "$HAS_YQ" = true ]; then
      # Safely update yaml key in-place using yq
      yq eval -i ".pipeline.label_path = \"${NEW_VALUE}\"" "$file"
      echo "  [yq] Updated: $(basename "$file")"
    else
      # Fallback using sed (handles nested/indented YAML keys)
      # Replaces 'label_path: ...' with the new value
      if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' -E "/^[[:space:]]*label_path:/s|:.*|: \"${NEW_VALUE}\"|" "$file"
      else
        sed -i -E "/^[[:space:]]*label_path:/s|:.*|: \"${NEW_VALUE}\"|" "$file"
      fi
      echo "  [sed] Updated: $(basename "$file")"
    fi
  done < <(find "$TARGET_PATH" -type f \( -name "*.yaml" -o -name "*.yml" \) -print0)

done

echo "Done."