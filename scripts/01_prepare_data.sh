#!/usr/bin/env bash
# Stage 1 of 4 — thesis Chapter 5.1 "Data Preparation".
#
# ATC ontology only: data/raw/ATC.csv -> data_atc.csv + data_atc_links.csv.
# The sales side belongs to Chapter 7 and is handled by stage 4, so that a reader without
# the licensed IQVIA export is never blocked here.
#
# Must run first: the output is sorted by ATC code, which is what makes every later
# artefact alignable by row position.
#
# Usage:  bash scripts/01_prepare_data.sh          (~1 min)

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LOG_DIR="logs/prepare/$STAMP"
mkdir -p "$LOG_DIR"

RAW_ATC="data/raw/ATC.csv"

echo "=== [1/4] data preparation — ATC ontology (Ch. 5.1)  ($STAMP) ==="
echo "python: $PY"
echo

require "$RAW_ATC" "Download the ATC export from BioPortal (version 2025_02_10) to $RAW_ATC."

run "atc_prepare" "$PY" src/data_prep/prepare_atc.py

echo "outputs: data/prepared_data/data_atc.csv, data_atc_links.csv"
echo "next   : bash scripts/02_build_embeddings.sh"
finish
