ISS(2,4) DUNE ND plotting code lives here.

Current implemented entry point:

```text
python plots/dune_nd/iss24/plot_fig4_dk2nu_iss24.py --point-id 16
python plots/dune_nd/iss24/plot_fig4_dk2nu_iss24.py --point-id 23
python plots/dune_nd/iss24/plot_iss24_scan_maps.py --workers 8 --chunk-size 900
```

It uses a saved `construct_24` point with `pmns_pass=1` and `eta_pass=1`,
reads the `U5_solver` complex mixing matrix, averages probabilities over the
dk2nu ND source profile, and writes:

```text
figures/dune_nd/iss24/construct24_point16/fig4/
figures/dune_nd/iss24/construct24_point23/fig4/
data/dune_nd/iss24/construct24_point16/fig4/
data/dune_nd/iss24/construct24_point23/fig4/
figures/dune_nd/iss24/scan_maps/
data/dune_nd/iss24/scan_maps/
```
