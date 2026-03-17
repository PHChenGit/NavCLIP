#!/usr/bin/env bash
# download_dataset.sh — NavCLIP dataset downloader

set -uo pipefail

# Always run from the project root so relative paths (e.g. ./datasets) are correct
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_URL="http://vision.ee.ccu.edu.tw/dataset/UAV_Taipei"

# Parallel arrays
DATASET_FILES=(
    "Daan_park_100k.zip"
    "Daan_park.zip"
    "NTU_playground_1M.zip"
    "NTU_playground_100k.zip"
    "NTU_playground_Cross_Season_100k.zip"
    "NTU_playground.zip"
    "pohsun_datasets_v1.zip"
    "satellites.zip"
    "Taichung_station_100k.zip"
    "taipei_gallery_1M.zip"
    "UCLA_Cross_Season_100k_archive.zip"
)
DATASET_LABELS=(
    "Daan Park 100K"
    "Daan Park"
    "NTU Playground 1M"
    "NTU Playground 100K"
    "NTU Playground Cross Season 100K"
    "NTU Playground"
    "Pohsun Dataset v1"
    "Satellites Dataset"
    "Taichung Station 100K"
    "Taipei Gallery 1M"
    "UCLA Cross Season 100K"
)
# Sizes in MB 
DATASET_SIZES_MB=(
    11182   # 10.92 GB  Daan Park 100K
    3963    #  3.87 GB  Daan Park
    34938   # 34.12 GB  NTU Playground 1M
    11120   # 10.86 GB  NTU Playground 100K
    12083   # 11.80 GB  NTU Playground Cross Season 100K
    3461    #  3.38 GB  NTU Playground
    16261   # 15.88 GB  Pohsun Dataset v1
    154     #  0.15 GB  Satellites Dataset
    15524   # 15.16 GB  Taichung Station 100K
    30597   # 29.88 GB  Taipei Gallery 1M
    66201   # 64.65 GB  UCLA Cross Season 100K
)

# Defaults
YES_FLAG=0
OUTPUT_DIR="${OUTPUT_DIR:-./datasets}"

# ===== Usage =====
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [OUTPUT_DIR]

Downloads NavCLIP datasets from the project server.

Arguments:
  OUTPUT_DIR   Destination directory (default: ./datasets)
               Can also be set via the OUTPUT_DIR environment variable.

Options:
  -y, --yes    Skip prompts: select all datasets without confirmation
  -h, --help   Show this message

Examples:
  $(basename "$0")                   # interactive, saves to ./datasets
  $(basename "$0") ~/data/navclip    # interactive, saves to ~/data/navclip
  $(basename "$0") -y /mnt/storage   # download everything, no prompts
  OUTPUT_DIR=/mnt/storage $(basename "$0") -y
EOF
    exit 0
}

# ===== Argument parsing =====
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)  YES_FLAG=1; shift ;;
        -h|--help) usage ;;
        -*)        echo "Unknown option: $1" >&2; usage ;;
        *)         OUTPUT_DIR="$1"; shift ;;
    esac
done

# ===== Helpers =====
die() { echo "ERROR: $*" >&2; exit 1; }

format_mb() {
    local mb=$1
    if (( mb >= 1024 )); then
        awk "BEGIN { printf \"%.1f GB\", $mb / 1024 }"
    else
        echo "${mb} MB"
    fi
}

available_space_mb() {
    df -k "$1" 2>/dev/null | awk 'NR==2 { printf "%d", $4 / 1024 }'
}

check_deps() {
    local missing=()
    for cmd in wget df awk; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    [[ ${#missing[@]} -eq 0 ]] || die "Required commands not found: ${missing[*]}"
}

check_deps

NUM_DATASETS=${#DATASET_FILES[@]}
SELECTED=()

# ===== Dataset selection =====
if [[ $YES_FLAG -eq 1 ]]; then
    for (( i=0; i<NUM_DATASETS; i++ )); do SELECTED+=("$i"); done
else
    echo "Available datasets:"
    echo ""
    for (( i=0; i<NUM_DATASETS; i++ )); do
        printf "  %2d) %-52s %s\n" \
            $((i+1)) \
            "${DATASET_LABELS[$i]}" \
            "($(format_mb "${DATASET_SIZES_MB[$i]}"))"
    done
    echo ""

    while true; do
        printf "Enter numbers to download (e.g. 1 3 5), or 'a' for all: "
        read -r SELECTION

        if [[ "$SELECTION" =~ ^[Aa]$ ]]; then
            for (( i=0; i<NUM_DATASETS; i++ )); do SELECTED+=("$i"); done
            break
        fi

        VALID=1
        TEMP_SELECTED=()
        for token in $SELECTION; do
            if [[ "$token" =~ ^[0-9]+$ ]] && (( token >= 1 && token <= NUM_DATASETS )); then
                TEMP_SELECTED+=("$((token - 1))")
            else
                echo "  Invalid selection: '$token'" >&2
                VALID=0
                break
            fi
        done

        if [[ $VALID -eq 1 && ${#TEMP_SELECTED[@]} -gt 0 ]]; then
            SELECTED=("${TEMP_SELECTED[@]}")
            break
        elif [[ $VALID -eq 1 ]]; then
            echo "  Please select at least one dataset."
        fi
    done
fi

[[ ${#SELECTED[@]} -gt 0 ]] || die "No datasets selected."

# ===== Size summary =====
TOTAL_MB=0
for idx in "${SELECTED[@]}"; do
    TOTAL_MB=$(( TOTAL_MB + DATASET_SIZES_MB[idx] ))
done

mkdir -p "$OUTPUT_DIR" || die "Cannot create output directory: $OUTPUT_DIR"

AVAIL_MB=$(available_space_mb "$OUTPUT_DIR")
MOUNT=$(df -k "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2{print $NF}')

echo ""
echo "Selected datasets:"
for idx in "${SELECTED[@]}"; do
    printf "  - %-52s %s\n" "${DATASET_LABELS[$idx]}" "($(format_mb "${DATASET_SIZES_MB[$idx]}"))"
done
echo ""
printf "  Total estimated size : %s\n" "$(format_mb $TOTAL_MB)"
printf "  Available disk space : %s  (on %s)\n" "$(format_mb $AVAIL_MB)" "$MOUNT"
printf "  Output directory     : %s\n" "$OUTPUT_DIR"
echo ""

if (( AVAIL_MB < TOTAL_MB )); then
    echo "WARNING: Estimated download size exceeds available disk space." >&2
    echo ""
fi

if [[ $YES_FLAG -eq 0 ]]; then
    read -r -p "Proceed with download? [y/N] " CONFIRM
    [[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
    echo ""
fi

# ===== Download =====
FAILED=()
COUNT=0
TOTAL_SELECTED=${#SELECTED[@]}

for idx in "${SELECTED[@]}"; do
    COUNT=$(( COUNT + 1 ))
    FILE="${DATASET_FILES[$idx]}"
    LABEL="${DATASET_LABELS[$idx]}"
    URL="${BASE_URL}/${FILE}"
    DEST="${OUTPUT_DIR}/${FILE}"
    TMP="${DEST}.tmp"

    echo "[$COUNT/$TOTAL_SELECTED] $LABEL  ($(format_mb "${DATASET_SIZES_MB[$idx]}"))"
    echo "  → $DEST"

    if [[ -f "$DEST" ]]; then
        echo "  Already exists, skipping. (Delete the file to re-download.)"
        echo ""
        continue
    fi

    # Download to a temp file; resume if a previous attempt was interrupted.
    # On success: rename to final destination.
    # On failure: remove the temp file so it is not mistaken for a complete file.
    if wget --continue \
            --show-progress \
            --tries=5 \
            --retry-connrefused \
            --timeout=60 \
            --waitretry=15 \
            -O "$TMP" \
            "$URL"; then
        mv "$TMP" "$DEST"
        echo "  Done."
    else
        rm -f "$TMP"
        echo "  FAILED: $FILE" >&2
        FAILED+=("${DATASET_LABELS[$idx]}")
    fi

    echo ""
done

# ===== Final report =====
echo "========================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All downloads completed successfully."
    echo "Files saved to: $OUTPUT_DIR"
    exit 0
else
    echo "Completed with ${#FAILED[@]} failure(s):"
    for label in "${FAILED[@]}"; do
        echo "  - $label"
    done
    echo ""
    echo "Re-run this script to retry failed downloads."
    exit 1
fi
