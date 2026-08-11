# Gallery — radiation

Examples validating the grey two-stream radiation schemes against Isca. Return to
the {doc}`index`. The physics is described in {doc}`../physics/radiation`.

(gallery-gray-frierson)=
## Grey radiation — Frierson scheme

:::{admonition} `scripts/plot_two_stream_gray_rad.py` · fixture
:class: note
Reads `tests/fixtures/two_stream_gray_rad_reference.npz`.
:::

Validates the Frierson grey-radiation two-stream scheme
(`rad_scheme='frierson'`) — prescribed longwave optical depth, no explicit
humidity feedback — against Isca. The left panel is the radiative heating profile
of a tropical column (jsca over Isca), the middle panel the surface downward
SW/LW fluxes vs latitude, and the right panel a jsca-vs-Isca scatter of the
heating rate at all points.

```{figure} ../figures/two_stream_gray_rad.png
:alt: Frierson grey radiation, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_two_stream_gray_rad.py
:language: python
:caption: scripts/plot_two_stream_gray_rad.py
```

(gallery-gray-byrne)=
## Grey radiation — Byrne & O'Gorman scheme

:::{admonition} `scripts/plot_two_stream_gray_rad_byrne.py` · fixture
:class: note
Reads `tests/fixtures/two_stream_gray_rad_byrne_reference.npz`.
:::

Validates the Byrne & O'Gorman (2013) grey scheme (`rad_scheme='byrne'`), whose
longwave optical depth depends on specific humidity, giving a water-vapour
greenhouse feedback absent from Frierson. The middle panel contrasts the two
schemes' surface downward LW vs latitude — Byrne's humidity dependence produces
more downward LW in the moist tropics.

```{figure} ../figures/two_stream_gray_rad_byrne.png
:alt: Byrne grey radiation, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_two_stream_gray_rad_byrne.py
:language: python
:caption: scripts/plot_two_stream_gray_rad_byrne.py
```
