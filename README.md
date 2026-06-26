# alpha_continuum

Spectral continuum normalization using an alpha-shape style method.

## Author

Mingjie Jian

## Method Background

Continuum normalisation is performed using the "alpha-roll" method described by
Xu et al. (2019) and Cretignier et al. (2020). In this method, a circle with
radius `alpha` is rolled along the top of the spectrum, and the contact points
between the circle and the flux curve are selected as continuum points. In this
implementation, `alpha` is adaptively determined from the difference between
the smoothed spectrum and the observed spectrum, so the radius becomes larger
in regions affected by absorption lines.

- Xu et al. 2019: [ADS](https://ui.adsabs.harvard.edu/abs/2019AJ....157..243X/abstract)
- Cretignier et al. 2020: [ADS](https://ui.adsabs.harvard.edu/abs/2020A&A...640A..42C/abstract)

## Main API

This package is mainly used through `alpha_continuum.normalization`.

```python
from alpha_continuum import normalization

spec_out = normalization(spec_in)
```

## Input Format

`spec_in` must be a `pandas.DataFrame` containing:

- `wave`: wavelength array
- `flux`: flux array

## `normalization(...)`

```python
normalization(
    spec_in,
    stretch=True,
    fit_method="poly",
    rollmax_width=20,
    base_ratio=2,
    penalty_ratio=1,
    max_radius_ratio=0.1,
    radius_override=None,
    force_edge_anchors=False,
    poly_deg=8,
    spline_s=None,
    printout=False,
    plot=False,
    plot_save_dir=None,
    detail_out=False,
    plot_title="",
)
```

Continuum-normalize a spectrum and return a new `DataFrame`.

### Parameters

- `spec_in`: input spectrum as a `pandas.DataFrame`. Must contain `wave` and `flux`.
- `stretch`: whether to rescale the flux axis internally so its span is comparable to the wavelength axis when selecting continuum edge points.
- `fit_method`: continuum interpolation method. Supported values are `"poly"`, `"spline"`, and `"akima"`.
- `rollmax_width`: rolling window width, in Angstrom, used when estimating line width and alpha radius.
- `base_ratio`: base scaling applied to the alpha radius.
- `penalty_ratio`: extra scaling applied to the alpha radius in deeper absorption regions.
- `max_radius_ratio`: maximum alpha radius as a fraction of the wavelength span of the current order. Set to `None` to disable this safeguard.
- `radius_override`: manually override the alpha radius. Pass a scalar to use one radius for the whole order, or an array with one radius per pixel. When set, the automatic FWHM-based radius logic is skipped.
- `force_edge_anchors`: force a right-edge continuum anchor using the same rough-continuum proxy used for the left-edge starting anchor. The left edge already has an anchor by default; this option mainly adds the symmetric right-edge constraint.
- `poly_deg`: polynomial degree used when `fit_method="poly"`.
- `spline_s`: smoothing factor used when `fit_method="spline"`.
- `printout`: whether to print progress information such as the estimated line FWHM.
- `plot`: whether to generate diagnostic plots.
- `plot_save_dir`: directory used to save diagnostic plots when `plot=True`.
- `detail_out`: whether to keep intermediate columns in the output. If `False`, only the original input columns plus `continuum` and `flux_normed` are returned.
- `plot_title`: title used in generated plots.

### Returns

Returns a `pandas.DataFrame`.

When `detail_out=False`, the result contains:

- original input columns
- `continuum`: estimated continuum
- `flux_normed`: normalized flux

When `detail_out=True`, intermediate processing columns are also included.

## Citation

If you use `alpha_continuum` in published research, please cite:

- Jian et al. 2026: [ADS](https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f1797J/abstract)

The alpha-roll method implemented in this package builds on the approaches
described by Xu et al. (2019) and Cretignier et al. (2020). These papers may
also be cited when discussing the methodological background in detail.

## Example

```python
import pandas as pd
from alpha_continuum import normalization

spec_in = pd.DataFrame({
    "wave": wave,
    "flux": flux,
})

spec_out = normalization(
    spec_in,
    fit_method="akima",
    plot=True,
    plot_save_dir="plots",
)
```
