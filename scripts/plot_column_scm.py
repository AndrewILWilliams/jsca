"""Diagnostic figure for the single-column model (SCM) PR.

Runs jsca's single-column model (:mod:`jsca.model.column`) from the Isca cold start
to radiative-convective equilibrium and plots the spin-up. This is a *behaviour*
demonstration of the physics-chain assembly (there is no Isca reference to overplot
yet -- that is issue-tracked follow-up work); it shows the SCM is stable and settles
into a sensible tropical single-column climate.

Config: a single column at the Frierson global-average latitude, the 25-level
Frierson sigma coordinate, Frierson grey-radiation + simple-Betts-Miller physics,
dt = 1440 s (the canonical column_test.py timestep). ~400 days of spin-up.

Run: python scripts/plot_column_scm.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401  (enables x64)
from jsca.dycore.press_and_geopot import pressure_variables
from jsca.model import column as C

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures" / "column_scm_spinup.png"

DT = 1440.0
DAYS = 400
STEPS = int(DAYS * 86400 / DT)
SAMPLE_EVERY = 30  # steps between time-series samples (~12 h)

m = C.build_column(dt=DT)
s0 = C.initial_state(m)

# reference full-level pressures (ps is fixed in the SCM) for the profile y-axis
_, _, p_full_1d, _ = pressure_variables(
    m.pk, m.bk, jnp.asarray(101325.0), m.vert_difference_option)
p_full_hpa = np.asarray(p_full_1d).ravel() / 100.0


def sample(state):
    """Scalars tracked over the spin-up (single column, so .ravel()[0])."""
    u, v, tg, qg, ps, t_surf = state
    t_air = tg[..., -1, 1]                 # near-surface air temperature (current)
    # column-integrated precipitable water: sum q * dp / g
    ph = m.pk + m.bk * ps[..., None]
    dp = ph[..., 1:] - ph[..., :-1]
    cwv = jnp.sum(qg[..., 1] * dp, axis=-1) / 9.80
    return jnp.stack([t_surf.ravel()[0], t_air.ravel()[0], cwv.ravel()[0]])


# cold-start forward step, then leapfrog with periodic sampling inside lax.scan
state = jax.jit(lambda s: C.step(m, s, m.dt))(s0)


def body(carry, _):
    s = carry
    def chunk(ss, _):
        s2, precip = C._step_full(m, ss)
        return s2, precip.ravel()[0]
    s, precips = jax.lax.scan(chunk, s, None, length=SAMPLE_EVERY)
    return s, jnp.concatenate([sample(s), jnp.array([jnp.mean(precips)])])


n_samples = STEPS // SAMPLE_EVERY
state, series = jax.lax.scan(jax.jit(body), state, None, length=n_samples)
series = np.asarray(series)
t_days = (np.arange(n_samples) + 1) * SAMPLE_EVERY * DT / 86400.0

t_surf_ts, t_air_ts, cwv_ts = series[:, 0], series[:, 1], series[:, 2]
precip_mmday = series[:, 3] * 86400.0     # kg/m^2/s -> mm/day

# final vs initial vertical profiles
_, _, tg_f, qg_f, _, _ = state
T_final = np.asarray(tg_f[..., 1]).ravel()
q_final = np.asarray(qg_f[..., 1]).ravel() * 1e3   # g/kg
T_init = np.asarray(s0[2][..., 1]).ravel()
q_init = np.asarray(s0[3][..., 1]).ravel() * 1e3

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(
    "jsca single-column model: spin-up to radiative-convective equilibrium\n"
    f"Frierson physics, 25 levels, lat={C.GLOBAL_AVERAGE_LAT_DEG:.1f}°, "
    f"dt={DT:.0f}s, {DAYS} days",
    fontsize=12)

a = ax[0, 0]
a.plot(t_days, t_surf_ts, label="surface (slab SST)", color="tab:red")
a.plot(t_days, t_air_ts, label="near-surface air", color="tab:orange")
a.set_xlabel("day"); a.set_ylabel("temperature (K)")
a.set_title("Temperature spin-up"); a.legend(); a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(t_days, precip_mmday, color="tab:blue")
a.set_xlabel("day"); a.set_ylabel("precipitation (mm/day)")
a.set_title("Precipitation"); a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(T_init, p_full_hpa, "--", color="grey", label="initial (264 K)")
a.plot(T_final, p_full_hpa, color="tab:red", label="equilibrium")
a.invert_yaxis(); a.set_xlabel("temperature (K)"); a.set_ylabel("pressure (hPa)")
a.set_title("Temperature profile"); a.legend(); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(q_init, p_full_hpa, "--", color="grey", label="initial (1 g/kg)")
a.plot(q_final, p_full_hpa, color="tab:green", label="equilibrium")
a.invert_yaxis(); a.set_xlabel("specific humidity (g/kg)"); a.set_ylabel("pressure (hPa)")
a.set_title("Humidity profile (q_decrease_only on)"); a.legend(); a.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.94))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
print(f"equilibrium: SST={t_surf_ts[-1]:.2f} K, near-surface air={t_air_ts[-1]:.2f} K, "
      f"precip={precip_mmday[-1]:.2f} mm/day, CWV={cwv_ts[-1]:.2f} kg/m^2")
