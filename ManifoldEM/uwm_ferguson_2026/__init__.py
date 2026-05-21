"""
UW-Milwaukee 2026 Ferguson analysis.

This subpackage is a verbatim copy of the Ferguson_plot_Python package
provided by Umeshika Dissanayaka (UW-Milwaukee, April 2026), based on
earlier work by Laura Williams and Russell Fung (UW-Milwaukee, 2018).

Only one mechanical change has been made to the original sources, i.e., 
intra-package imports of the form ``from foo_ import bar`` have been
changed to relative imports ``from .foo_ import bar`` so that the
modules can be loaded as a Python subpackage. No numerical or
algorithmic code has been altered.

See ``ATTRIBUTION.md`` for full provenance and license context.

Public API
----------
- ``A_ij(Dsq, sigma)``           — Gaussian kernel sum at given σ
- ``fit_ramp(x, y, tol, p)``     — find central linear ramp of a sigmoid
- ``sigma_of_interest(Dsq)``     — data-adaptive σ grid
- ``linear_regression(x, y)``    — basic OLS fit (slope, intercept)
- ``analyze(h5_file)``           — full pipeline on an H5 distance file
- ``ferguson_analysis(h5_file)`` — top-level entry point

The wrapper that adapts this package to ManifoldEM's ``fergusonE``
contract lives in ``ManifoldEM._ferguson_uwm`` (one level up).
"""
from .A_ij_ import A_ij
from .fit_ramp_ import fit_ramp
from .sigma_of_interest_ import sigma_of_interest
from .linear_regression_ import linear_regression
from .analyze_ import analyze
from .run_ferguson_ import ferguson_analysis

__all__ = [
    "A_ij",
    "fit_ramp",
    "sigma_of_interest",
    "linear_regression",
    "analyze",
    "ferguson_analysis",
]
