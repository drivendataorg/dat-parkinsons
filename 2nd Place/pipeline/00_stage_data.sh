#!/usr/bin/env bash
# 00_stage_data.sh -- put the DrivenData training download where the code expects it.
#
#   data/raw/niftis/<uid>.nii.gz   (1362 training scans; the zip holds them at its root)
#   data/raw/labels.csv            (header uid,is_pathologic; a byte copy of the train labels CSV)
#
# The original project kept the same two items in data-demo/ (the unpack command
# itself was not recorded). src/ reads DAT_DATA_DIR if set, else data/raw.
#
# Idempotent: when data/raw already holds 1362 scans and a 1362-row labels.csv, it
# only re-checks counts. It never prints file names, uids or labels, only counts
# and hashes.
#
# Usage:
#   bash pipeline/00_stage_data.sh <training niftis .zip> <training labels .csv>
#   bash pipeline/00_stage_data.sh                  # verify an existing staging only
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${DAT_DATA_DIR:-$ROOT/data/raw}"
ZIP="${1:-}"
LABELS="${2:-}"
N_EXPECTED=1362
# sha256 of the files downloaded for the competition (for information only)
ZIP_SHA=87a17ccd9eeb44ea24dab23eefd9a56eabe92c95992ba04c678ba0270b3230bd
LABELS_SHA=1660731b5d51b1f91b021c41779478722139b487d44a3e25b7f54800c16ef75e

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
n_scans() { find "$RAW/niftis" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l | tr -d ' '; }
n_rows() { [ -f "$RAW/labels.csv" ] && echo $(( $(wc -l < "$RAW/labels.csv") )) || echo 0; }
mkdir -p "$RAW/niftis"

# scans
if [ "$(n_scans)" -eq "$N_EXPECTED" ]; then
  echo "[00_stage] niftis already staged ($(n_scans) files), not unzipping"
else
  [ -n "$ZIP" ] || { echo "[00_stage] $(n_scans) scans in $RAW/niftis; pass the niftis zip as argument 1" >&2; exit 1; }
  if [ "$(sha256 "$ZIP")" = "$ZIP_SHA" ]; then echo "[00_stage] zip sha256 matches the competition download"
  else echo "[00_stage] WARNING: zip sha256 differs from the original competition download"; fi
  unzip -q -o -j "$ZIP" '*.nii.gz' -d "$RAW/niftis"   # -j: flatten, files land at niftis/<uid>.nii.gz
fi

# labels
if [ "$(n_rows)" -eq $((N_EXPECTED + 1)) ] && [ -z "$LABELS" ]; then
  echo "[00_stage] labels.csv already staged"
elif [ -n "$LABELS" ]; then
  if [ "$(sha256 "$LABELS")" = "$LABELS_SHA" ]; then echo "[00_stage] labels sha256 matches the competition download"
  else echo "[00_stage] WARNING: labels sha256 differs from the original competition download"; fi
  cmp -s "$LABELS" "$RAW/labels.csv" 2>/dev/null || cp "$LABELS" "$RAW/labels.csv"
else
  echo "[00_stage] $RAW/labels.csv missing or incomplete; pass the labels csv as argument 2" >&2; exit 1
fi

# verification (counts only)
hdr="$(head -1 "$RAW/labels.csv" | tr -d '\r')"
[ "$hdr" = "uid,is_pathologic" ] || { echo "[00_stage] unexpected labels header: $hdr" >&2; exit 1; }
missing=0
while IFS=, read -r uid _; do
  [ -f "$RAW/niftis/$uid.nii.gz" ] || missing=$((missing + 1))
done < <(tail -n +2 "$RAW/labels.csv" | tr -d '\r')
echo "[00_stage] scans: $(n_scans)  label rows: $(( $(n_rows) - 1 ))  labelled uids without a scan: $missing"
echo "[00_stage] labels.csv sha256 $(sha256 "$RAW/labels.csv" | cut -c1-16)..."
[ "$(n_scans)" -eq "$N_EXPECTED" ] && [ "$(( $(n_rows) - 1 ))" -eq "$N_EXPECTED" ] && [ "$missing" -eq 0 ] \
  || { echo "[00_stage] staging incomplete" >&2; exit 1; }
echo "[00_stage] OK"
