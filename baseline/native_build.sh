#!/usr/bin/env bash
# Phase-0: build the pinned Isca Fortran baseline natively (no Docker daemon
# needed), following the recipe in the repo's own Dockerfile. Run on the
# machine that will produce baseline timings and golden fixtures.
#
# Usage: ISCA_SRC=/path/to/Isca-master ./native_build.sh
set -euo pipefail

ISCA_SRC="${ISCA_SRC:?set ISCA_SRC to the Isca checkout (the unzipped Iscamaster snapshot)}"

# --- system packages (from Isca's Dockerfile) --------------------------------
sudo_maybe() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }
sudo_maybe apt-get update
DEBIAN_FRONTEND=noninteractive sudo_maybe apt-get install -y \
    build-essential gfortran git \
    libnetcdf-dev libnetcdff-dev libhdf5-openmpi-dev libopenmpi-dev openmpi-bin tcl

# --- python front end ---------------------------------------------------------
pip install -r "$ISCA_SRC/src/extra/python/requirements.txt"
pip install -e "$ISCA_SRC/src/extra/python"

# --- environment (Dockerfile uses GFDL_ENV=docker) ----------------------------
export GFDL_BASE="$ISCA_SRC"
export GFDL_ENV=docker
export GFDL_WORK="${GFDL_WORK:-$HOME/isca_work}"
export GFDL_DATA="${GFDL_DATA:-$HOME/isca_data}"
mkdir -p "$GFDL_WORK" "$GFDL_DATA"

cat <<EOF
Environment ready. Persist in your shell profile:
  export GFDL_BASE=$GFDL_BASE
  export GFDL_ENV=docker
  export GFDL_WORK=$GFDL_WORK
  export GFDL_DATA=$GFDL_DATA
Then run:  python baseline/run_timing_case.py --case held_suarez --cores 1 4 8 16
EOF
