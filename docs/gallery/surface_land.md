# Gallery — mixed layer, land & bucket

Examples validating the slab-ocean surface and land hydrology against Isca.
Return to the {doc}`index`. The physics is described in {doc}`../physics/surface`.

(gallery-mixed-layer)=
## Slab-ocean mixed layer

:::{admonition} `scripts/plot_mixed_layer.py` · fixture
:class: note
Reads `tests/fixtures/mixed_layer_reference.npz`.
:::

Validates the slab-ocean `mixed_layer` surface energy balance — the implicit
coupling of surface fluxes to a fixed-depth ocean heat capacity that sets SST.
The left panel shows the per-step SST increment across the test grid (the closed
implicit balance of surface fluxes against slab-ocean heat capacity); the right
panel is a jsca-vs-Isca scatter of the SST update and the corrected lowest-level
$T/q$ increments the `vert_diff` up-sweep consumes.

```{figure} ../figures/mixed_layer.png
:alt: Mixed layer SST update, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_mixed_layer.py
:language: python
:caption: scripts/plot_mixed_layer.py
```

(gallery-bucket)=
## Manabe bucket hydrology

:::{admonition} Fixture: `tests/test_bucket_fixtures.py` · fixture
:class: note
The soil-moisture stepping is validated to `rtol` 1e-13 against Isca in
`tests/test_bucket_fixtures.py`, reading
`tests/fixtures/bucket_stepping_reference.npz`. The figure below was produced in
the bucket-hydrology pull request; the code shown is the fixture test that gates
the step.
:::

Validates the Manabe bucket soil-moisture reservoir step
({py:func}`jsca.physics.bucket.bucket_step`) — prognostic land water updated by
precipitation minus evaporation with a capacity cap and runoff.

```{figure} ../figures/bucket_stepping.png
:alt: Bucket soil-moisture stepping, jsca vs Isca
```

```{literalinclude} ../../tests/test_bucket_fixtures.py
:language: python
:caption: tests/test_bucket_fixtures.py
```
