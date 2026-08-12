# Radiation & insolation

`jsca` treats the atmosphere as a **grey** (wavelength-independent) absorber with
a prescribed optical-depth profile — no spectral bands, no clouds. Longwave and
shortwave are independent two-stream problems. This is the radiation of the
Frierson idealized moist model, with an optional humidity- and CO₂-dependent
variant.

:::{admonition} Source
:class: note
{py:mod}`jsca.physics.two_stream_gray_rad` (ports
`two_stream_gray_rad.F90`); {py:mod}`jsca.physics.astronomy` (ports the
`diurnal_solar` path of `astronomy.f90`). **References:** Frierson, Held &
Zurita-Gotor (2006, 2007); Byrne & O'Gorman (2013).
:::

The step is split into a **down pass** (shortwave + downward longwave → surface
fluxes, computed *before* the surface temperature is known) and an **up pass**
(upward longwave + heating, using the *updated* surface temperature), so the
mixed-layer surface update slots between them — matching Isca.

## Shortwave

Shortwave is identical for both longwave schemes: a fixed top-of-atmosphere
insolation attenuated downward through an optical depth that grows as a power of
pressure. With normalized pressure $\hat p = p_{\text{half}}/p_{\text{std}}$ and
$p_2 = \tfrac14(1 - 3\sin^2\phi)$,

$$
\text{insol} = \tfrac14 S_0\big(1 + \delta_{\text{sol}}\,p_2 + \delta_{\text{sw}}\sin\phi\big),
$$
$$
\tau_{sw,0} = (1 - \text{sw\_diff}\,\sin^2\phi)\,\text{atm\_abs},
\qquad
\tau_{sw}(k) = \tau_{sw,0}\,\hat p(k)^{\text{solar\_exponent}},
$$
$$
\text{SW}^\downarrow(k) = \text{insol}\;e^{-\tau_{sw}(k)}.
$$

Upward shortwave is a column-constant reflection
$\text{SW}^\uparrow = \alpha\,\text{SW}^\downarrow(\text{sfc})$, and the surface
absorbs $(1-\alpha)\,\text{SW}^\downarrow(\text{sfc})$.

## Longwave — two-stream grey integration

With source function $b = \sigma T^4$, the downward pass runs top → surface and
the upward pass surface → top:

$$
\text{LW}^\downarrow(k{+}1) = \text{LW}^\downarrow(k)\,\text{dtrans}(k)
    + b(k)\,\big(1 - \text{dtrans}(k)\big),
\qquad \text{LW}^\downarrow(0) = 0,
$$
$$
\text{LW}^\uparrow(k) = \text{LW}^\uparrow(k{+}1)\,\text{dtrans}(k)
    + b(k)\,\big(1 - \text{dtrans}(k)\big),
\qquad \text{LW}^\uparrow(K) = \sigma T_{\text{surf}}^4,
$$

where the **layer transmissivity** is
$\text{dtrans}(k) = \exp\!\big(-(\tau_{lw}(k{+}1) - \tau_{lw}(k))\big)$. The two
longwave schemes differ **only** in the optical depth.

### `rad_scheme='frierson'` (default)

Optical depth prescribed in latitude and pressure, mixing a linear
(well-mixed-gas) and a power-law (water-vapour-like) term:

$$
\tau_{lw,0} = \big(\tau_{\text{eq}} + (\tau_{\text{pole}} - \tau_{\text{eq}})\sin^2\phi\big)\,\text{odp},
$$
$$
\tau_{lw}(k) = \tau_{lw,0}\left[\text{linear\_tau}\,\hat p(k)
    + (1 - \text{linear\_tau})\,\hat p(k)^{\text{wv\_exponent}}\right].
$$

There is no explicit humidity dependence — the water-vapour greenhouse is baked
into the fixed profile.

### `rad_scheme='byrne'` (Byrne & O'Gorman 2013)

The optical depth grows with **specific humidity and CO₂**, giving a genuine
water-vapour feedback. The per-layer increment (with $\Delta p$ the layer
thickness) is

$$
\Delta\tau = \left(a\,\mu + 0.17\ln\frac{\text{CO}_2}{360} + b\,q\right)
    \frac{\Delta p}{p_{\text{std,Earth}}},
\qquad
\text{dtrans} = e^{-\Delta\tau},
$$

with defaults $a = 0.8678$ (`bog_a`), $b = 1997.9$ (`bog_b`), $\mu = 1$
(`bog_mu`). The absorption coefficients are non-dimensionalized by the **Earth**
surface pressure regardless of `pstd_mks`, and the scheme requires the specific
humidity $q$ to be passed in.

:::{admonition} Not yet ported
:class: note
The Geen (2015) two-band window scheme and the Schneider & Liu (2009)
giant-planet scheme are on the roadmap but not yet implemented.
:::

## Radiative heating

From the net (positive-up) flux divergence:

$$
F_{\text{rad}} = \big(\text{LW}^\uparrow - \text{LW}^\downarrow\big)
              + \big(\text{SW}^\uparrow - \text{SW}^\downarrow\big),
$$
$$
\left(\frac{\partial T}{\partial t}\right)_{\text{rad}}(k)
  = \text{diabatic\_acce}\cdot
    \frac{g\,\big[F_{\text{rad}}(k{+}1) - F_{\text{rad}}(k)\big]}
         {c_p\,\big[p_{\text{half}}(k{+}1) - p_{\text{half}}(k)\big]}.
$$

## Seasonal & diurnal insolation (`do_seasonal`)

By default the shortwave uses a fixed perpetual-equinox annual-mean profile. With
`do_seasonal=True`, the insolation is instead computed astronomically from the
ported `diurnal_solar`, which returns the cosine of the solar zenith angle (or
its analytic time-average over a radiation step), the daylight fraction, and the
Earth–Sun distance factor $rrsun = (a/r)^2$; the insolation is then $S_0\cos z$.

The pipeline:

- **Orbital-angle table** (`build_orbit_angle`): RK4-integrate
  $d(\text{angle})/dt = \text{norm}\cdot r^{-2}(\text{angle})$ (Kepler's second
  law, $\text{norm} = \sqrt{1 - e^2}$) from the autumnal equinox. For the default
  circular orbit ($e = 0$) the table is exactly linear and $rrsun \equiv 1$.
- **Distance factor:** $r = (1 - e^2)/(1 + e\cos(\text{ang} - \varpi))$,
  $rrsun = r^{-2}$, with $\varpi$ the longitude of perihelion.
- **Declination:** $\delta = \arcsin(-\sin(\text{obliq})\sin(\text{ang}))$.
- **Half-day arc:** $h = \arccos(-\tan\phi\tan\delta)$, saturating to $\pi$
  (polar day) or $0$ (polar night).
- **Zenith angle:** with $a = \sin\phi\sin\delta$, $b = \cos\phi\cos\delta$ and
  local time $t = \text{gmt} + \text{lon} - \pi$,

  $$
  \cos z = \max\!\big(0,\; a + b\cos t\big) \quad (|t| < h),
  $$

  and when a step interval is given, $\cos z$ is the analytic true time-average
  over the interval.

`solday ≥ 0` freezes the orbital position at a chosen day-of-year — a perpetual
season retaining the diurnal cycle. The astronomy is fixture-validated to
$10^{-12}$, and the seasonal SCM is validated against Isca (see
{doc}`../validation/index`).

## Key configuration

```{list-table} `GrayRadParams` (selected)
:header-rows: 1
:widths: 28 20 52

* - Field
  - Default
  - Meaning
* - `rad_scheme`
  - `'frierson'`
  - Longwave scheme: `'frierson'` or `'byrne'`.
* - `solar_constant` $S_0$
  - 1360
  - Top-of-atmosphere solar constant (W m⁻²).
* - `del_sol`
  - 1.4
  - Meridional shortwave structure ($\delta_{\text{sol}}$).
* - `del_sw`
  - 0
  - Hemispheric shortwave asymmetry ($\delta_{\text{sw}}$).
* - `ir_tau_eq`, `ir_tau_pole`
  - 6.0, 1.5
  - Equator / pole longwave opacity (Frierson).
* - `atm_abs`
  - 0.2
  - Shortwave atmospheric absorption (Frierson override; namelist default 0).
* - `linear_tau`
  - 0.1
  - Linear/power-law optical-depth mix.
* - `wv_exponent`, `solar_exponent`
  - 4, 4
  - Pressure exponents for LW / SW optical depth.
* - `carbon_conc`
  - 360
  - CO₂ concentration (ppmv), used by `'byrne'`.
* - `do_seasonal`
  - `False`
  - Switch to astronomically computed insolation.
* - `solday`, `equinox_day`
  - −, 0.75
  - Freeze day-of-year; NH autumn equinox as fraction of year.
```
