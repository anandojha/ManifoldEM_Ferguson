# UWM 2026 Ferguson Analysis — Attribution

## Original authors

- **Laura Williams** & **Russell Fung** (UW-Milwaukee, 2018) — original
  implementation of `A_ij`, `fit_ramp`, `linear_regression`,
  `sigma_of_interest`, `read_h5`, `plot`.
- **Russell Fung** (UW-Milwaukee, 2018–2019, updated 2022) — `write_h5`,
  `report_runtime`, `run_ferguson_`.
- **Umeshika Dissanayaka** (UW-Milwaukee, 2026) — updates to `analyze_`
  and `plot_` (numerical-stability handling for log of small kernel
  sums; `dimensionality` annotation in plots).

Per-file copyright headers in each ``.py`` file preserve the original
attribution from upstream.

## Source

Provided by Umeshika Dissanayaka via the
``Ferguson_plot_Python.zip`` archive (April 2026), accompanied by the
``Running Ferguson Analysis.pdf`` user guide. The unmodified zip is
retained in the project history for reference.

## Modifications made when vendoring

The source files in this directory are unchanged from the upstream
distribution **except** for converting nine intra-package imports to
relative imports so the modules can be loaded as a Python subpackage:

| File                  | Original                                 | Vendored                                  |
| --------------------- | ---------------------------------------- | ----------------------------------------- |
| ``analyze_.py``       | ``from A_ij_ import A_ij`` (and 4 more)  | ``from .A_ij_ import A_ij`` (and 4 more)  |
| ``fit_ramp_.py``      | ``from linear_regression_ import …``     | ``from .linear_regression_ import …``     |
| ``run_ferguson_.py``  | ``from analyze_ import analyze`` (and 2 more)  | ``from .analyze_ import analyze`` (and 2 more)  |

No numerical, algorithmic, or interface changes have been made. The
``__init__.py`` and this ``ATTRIBUTION.md`` are added; nothing else is
modified.

## How this package is used in ManifoldEM

This subpackage provides the UWM 2026 Ferguson backend. The drop-in
adapter that exposes a ``fergusonE``-compatible interface (so callers
in ``DMembeddingII.py`` need not change) lives one directory up at
``ManifoldEM/_ferguson_uwm.py``.

## License / use

Distributed with permission from the UW-Milwaukee group for use within
the ManifoldEM project. For external redistribution or use outside this
project, contact the original authors directly.
