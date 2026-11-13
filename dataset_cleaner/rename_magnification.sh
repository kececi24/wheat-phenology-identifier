#!/usr/bin/env bash
# Update the final 1X/10X filename token to match its containing folder.
set -euo pipefail

if [[ $# -ne 1 || ! -d $1 ]]; then
    printf 'Usage: bash %s STATION_OR_DATASET_OR_MAGNIFICATION_FOLDER\n' "$0" >&2
    exit 1
fi

root=$(cd -- "$1" && pwd -P)
renamed=0
while IFS= read -r -d '' source; do
    directory=${source%/*}
    label=${directory##*/}
    label=${label^^}
    [[ $label == 1X || $label == 10X ]] || continue

    filename=${source##*/}
    extension=${filename##*.}
    stem=${filename%.*}
    # Preserve an existing collision suffix, for example photo-1X(2).jpeg.
    if [[ ! $stem =~ ^(.*[-_[:space:]])?(10|1)[xX]([[:space:]]*\([0-9]+\))?$ ]]; then
        continue
    fi
    [[ ${BASH_REMATCH[2]}X != "$label" ]] || continue
    base=${BASH_REMATCH[1]}${label}
    new_stem=${base}${BASH_REMATCH[3]}

    destination=$directory/$new_stem.$extension
    number=2
    while [[ -e $destination || -L $destination ]]; do
        destination=$directory/$base\($number\).$extension
        number=$((number + 1))
    done

    # -n also protects against a destination created after the existence check.
    mv -n -T -- "$source" "$destination"
    if [[ -e $source || -L $source ]]; then
        printf 'Skipped occupied destination: %s\n' "$destination" >&2
        continue
    fi
    printf '%s -> %s\n' "$source" "$destination"
    renamed=$((renamed + 1))
done < <(find "$root" \
    -type d \( -iname bad_images -o -iname truncated_images -o -iname corrupt_images \) -prune -o \
    -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
        -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) -print0)

printf 'Renamed %d image(s).\n' "$renamed"
