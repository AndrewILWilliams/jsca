"""Parse the FMS clock table from an Isca run logfile.

At the end of a run FMS prints a tabulated clock section, e.g.:

     Tabulating mpp_clock statistics across 4 PEs...
                                        tmin       tmax       tavg       ...
     Total runtime                 123.456789 123.456789 123.456789
     Main loop                     120.000000 121.000000 120.500000
     Transforms                     40.123456  ...

Use 'Main loop' tmax for per-step throughput (excludes start-up/teardown):
    steps_per_s = n_steps / main_loop_tmax

Usage:
    python parse_timings.py /path/to/logfile [--steps N]
Isca writes run logs under the experiment data directory (e.g.
$GFDL_DATA/<exp_name>/run0001/) and echoes them to stdout during exp.run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROW = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_ ()/&.-]*?)\s{2,}((?:[0-9.eE+-]+\s+){1,}[0-9.eE+-]+)\s*$")


LOGGER_PREFIX = re.compile(r"^.*? - isca - (?:DEBUG|INFO|WARNING) - ")


def parse_clock_table(text: str) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    in_table = False
    for line in text.splitlines():
        line = LOGGER_PREFIX.sub("", line)  # exp.run logs echo FMS output with a prefix
        if "Tabulating mpp_clock statistics" in line:
            in_table = True
            continue
        if in_table:
            m = ROW.match(line)
            if m:
                name = m.group(1).strip()
                try:
                    vals = [float(v) for v in m.group(2).split()]
                except ValueError:
                    continue
                rows[name] = vals
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", type=Path)
    ap.add_argument("--steps", type=int, default=None, help="model steps in the run")
    args = ap.parse_args()

    rows = parse_clock_table(args.logfile.read_text(errors="replace"))
    if not rows:
        raise SystemExit("no FMS clock table found — did the run complete?")
    for name, vals in rows.items():
        print(f"{name:<32s} {vals}")
    for key in ("Main loop", "Total runtime"):
        if key in rows and args.steps:
            tmax = max(rows[key][: min(2, len(rows[key]))])
            print(f"\n{key}: {tmax:.3f} s -> {args.steps / tmax:,.2f} steps/s")


if __name__ == "__main__":
    main()
