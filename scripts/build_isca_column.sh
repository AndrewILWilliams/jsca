#!/usr/bin/env bash
# Turnkey build of Isca's single-column model, for regenerating the SCM reference
# that tests/test_column_vs_isca.py validates against. Verified 2026-08 in a fresh
# sandbox with gfortran 13.3. Run from anywhere; set ISCA_DIR to relocate.
#
#   bash scripts/build_isca_column.sh
#   python scripts/run_isca_column_reference.py 40      # -> atmos_daily.nc
#   # (Isca's run.sh uses `mpirun -np 1`, which refuses to run as root; if the
#   #  wrapped run fails, run the built exe directly -- see run_isca_column_reference.py)
#   cp $GFDL_WORK/experiment/scm_reference/run/atmos_daily.nc \
#      baseline/reference/column_scm_isca_daily_t264.nc
#   python scripts/distill_column_reference.py           # -> committed .npz
#
# The committed .npz is what CI uses, so none of this runs in CI -- it is only for
# regenerating the golden reference when the physics config or pinned Isca changes.
set -euo pipefail

ISCA_DIR="${ISCA_DIR:-/tmp/isca}"
ISCA_COMMIT="a290bc376d84d0ee83adbb80eb374b9f629c3534"   # the pinned validation target

echo "[1/4] system toolchain (gfortran + netCDF + OpenMPI)"
if ! command -v gfortran >/dev/null; then
  apt-get update -qq || true
  apt-get install -y --fix-missing gfortran libnetcdf-dev libnetcdff-dev \
    libopenmpi-dev openmpi-bin
fi

echo "[2/4] pinned Isca checkout at $ISCA_DIR"
if [ ! -d "$ISCA_DIR/.git" ]; then
  git clone https://github.com/ExeClim/Isca "$ISCA_DIR"
fi
git -C "$ISCA_DIR" checkout "$ISCA_COMMIT"

echo "[3/4] patch the gfortran mkmf template for modern gfortran (>=10)"
TMPL="$ISCA_DIR/src/extra/python/isca/templates/mkmf.template.gfort"
if ! grep -q "fallow-argument-mismatch" "$TMPL"; then
  # old FMS trips argument-mismatch / BOZ errors that were warnings pre-gfortran-10
  sed -i 's/-fno-range-check \\/-fno-range-check -fallow-argument-mismatch -fallow-invalid-boz \\/' "$TMPL"
fi

echo "[4/4] python front-end deps"
pip install -q f90nml jinja2 sh

echo "done. Isca column model ready to build via ColumnCodeBase (see run_isca_column_reference.py)."
echo "  export GFDL_BASE=$ISCA_DIR GFDL_ENV=gfortran GFDL_WORK=/tmp/isca_work GFDL_DATA=/tmp/isca_data"
echo "  export PYTHONPATH=$ISCA_DIR/src/extra/python:\$PYTHONPATH"
