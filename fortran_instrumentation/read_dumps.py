"""Reader for jsca_dump_mod fixtures (raw float64 stream + .meta manifest).

Usage:
    from read_dumps import read_dump, read_all
    arr = read_dump("dumps/jsca_000001_sin_hem_pe000")      # numpy array, Fortran order
    fixtures = read_all("dumps")                             # {name: [arrays in call order]}

Arrays are returned with Fortran (column-major) axis order preserved, i.e.
``arr.shape`` equals the Fortran ``shape(x)``. Transpose at the comparison
site if the Python code uses a different layout — never mutate fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def read_dump(stem: str | Path) -> np.ndarray:
    stem = Path(stem)
    meta = (stem.parent / (stem.name + ".meta")).read_text().split("\n")
    dtype = meta[1].strip()
    dims = [int(v) for v in meta[2].split()][1:]
    data = np.fromfile(stem.parent / (stem.name + ".bin"), dtype=dtype)
    return data.reshape(dims, order="F")


def read_all(dump_dir: str | Path) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for meta in sorted(Path(dump_dir).glob("jsca_*.meta")):
        stem = meta.with_suffix("")
        name = meta.read_text().split("\n")[0].strip()
        out[name].append(read_dump(stem))
    return dict(out)
