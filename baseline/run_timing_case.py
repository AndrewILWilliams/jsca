"""Phase-0 baseline timing: run an Isca test case briefly at several core
counts and record wall time per model step (scoping doc §3.4).

Imports the shipped test-case script unmodified (safe: the run is under
``if __name__ == '__main__'``), so the configuration is exactly Isca's — only
the run length is shortened. Requires the isca package + GFDL_* env vars
(see native_build.sh).

Usage:
    python run_timing_case.py --case held_suarez --cores 1 4 8 16 --days 8

Wall time here includes model start-up; for per-step numbers prefer the FMS
clock table from the run's logfile (parse_timings.py) and use --days large
enough (>= 8) that start-up amortizes.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).parent
CASES = {
    "held_suarez": "held_suarez/held_suarez_test_case.py",
    "frierson": "frierson/frierson_test_case.py",
}


def load_case(case: str):
    base = Path(os.environ["GFDL_BASE"])
    path = base / "exp" / "test_cases" / CASES[case]
    spec = importlib.util.spec_from_file_location(f"tc_{case}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # guarded by __main__ in the test cases
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="held_suarez", choices=sorted(CASES))
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--days", type=int, default=8)
    args = ap.parse_args()

    mod = load_case(args.case)
    exp, cb = mod.exp, mod.cb
    cb.compile()

    exp.namelist["main_nml"]["days"] = args.days
    dt_atmos = exp.namelist["main_nml"].get("dt_atmos")
    steps = args.days * 86400 / dt_atmos if dt_atmos else None

    run_kwargs = {}
    if "overwrite_data" in inspect.signature(exp.run).parameters:
        run_kwargs["overwrite_data"] = True

    results = []
    for n in args.cores:
        t0 = time.time()
        exp.run(1, use_restart=False, num_cores=n, **run_kwargs)
        wall = time.time() - t0
        rec = {
            "case": args.case,
            "cores": n,
            "days": args.days,
            "dt_atmos": dt_atmos,
            "wall_s": round(wall, 3),
            "steps_per_s_incl_startup": round(steps / wall, 3) if steps else None,
        }
        print(rec)
        results.append(rec)

    out = HERE / "timings.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(existing + results, indent=2))
    print(f"appended to {out}")


if __name__ == "__main__":
    main()
