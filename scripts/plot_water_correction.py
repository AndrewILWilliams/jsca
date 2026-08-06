"""Figure for the water_correction PR: global humidity conservation, jsca vs Isca.

Left: for one column, the ratio of corrected to input humidity vs pressure — a
single constant factor below the 200 hPa limit (the rescaled region), and exactly
1 above it (high, thin levels left untouched by the MiMA pressure limit). Right:
jsca-vs-Isca scatter of the corrected humidity at every grid point.

Run: python scripts/plot_water_correction.py
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.dycore import water_correction
from jsca.grid.transforms import TransformParams

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "water_correction_reference.npz")


def _t3(a):
    return np.transpose(a, (1, 0, 2))


nlon = int(fx["wc_meta"][0])
params = TransformParams(
    legendre=None, legendre_wts=None, sin_lat=None,
    wts_lat=jnp.asarray(fx["wc_wts_lat"]),
    mask_prognostic=None, mask_storage=None, lap_eig=None, coeffs=None,
    nlon=nlon, num_fourier=0,
)
limit = float(fx["wc_limit"][0])
q_in = _t3(fx["wc_q_in"])
p_full = _t3(fx["wc_p_full"])
q_out, factor = water_correction(
    params, fx["wc_pk"], fx["wc_bk"], q_in, fx["wc_psg"].T, p_full,
    float(fx["wc_mean_water_prev"][0]), limit)
q_out = np.asarray(q_out)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

# left: correction ratio profile for one column
i, j = 0, 0
ratio = q_out[i, j] / q_in[i, j]
p_col = p_full[i, j] / 100.0   # hPa
ax = axes[0]
ax.plot(ratio, p_col, "-o", color="tab:blue", ms=5)
ax.axhline(limit / 100.0, color="tab:red", lw=0.9, ls="--",
           label=f"water_correction_limit ({limit / 100:.0f} hPa)")
ax.axvline(1.0, color="0.7", lw=0.8)
ax.set_ylim(p_col.max(), 0)
ax.set_xlabel("corrected / input humidity")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"water correction factor (i={i}, j={j})")
ax.legend(fontsize=8)

# right: jsca vs Isca at every point
ax = axes[1]
ref = _t3(fx["wc_q_out"])
ax.scatter(ref.ravel() * 1e3, q_out.ravel() * 1e3, s=12, alpha=0.4, color="tab:green")
lim = np.array([0, ref.max() * 1e3 * 1.05])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca corrected humidity (g/kg)")
ax.set_ylabel("jsca corrected humidity (g/kg)")
ax.set_title("all grid points")

maxabs = np.abs(q_out - ref).max()
fig.suptitle(
    "Global water (humidity) conservation correction: jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "water_correction.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
