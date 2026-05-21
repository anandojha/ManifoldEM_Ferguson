"""
fergusonE-compatible wrapper around the UWM 2026 Ferguson backend.

Purpose
-------
``ManifoldEM.core.fergusonE`` and ``ManifoldEM.uwm_ferguson_2026`` solve
the same problem (find the optimal Gaussian-kernel bandwidth from a
log-kernel-sum sigmoid) but with different conventions:

==========================  =========================  ===========================
Aspect                      ``core.fergusonE``         UWM 2026 backend
==========================  =========================  ===========================
Input format                ``D``  (distances, 1-D)    ``Dsq`` (squared distances)
Sweep variable              ``logEps``                 ``sigma`` (linear, not log)
Kernel form                 ``exp(-D**2 / eps)``       ``exp(-Dsq / sigma**2)``
Curve fit                   ``curve_fit`` to tanh      OLS on central ramp
σ_opt formula               ``sqrt(2*exp(-b/a))``      ``exp(x_mid)``
Bonus output                —                          dimensionality (slope)
==========================  =========================  ===========================

The math is consistent: ``sigma**2 == eps``, i.e. ``sigma = exp(logEps/2)``.

This wrapper accepts the current ``fergusonE`` call signature, runs the
UWM backend, and returns a 4-tuple in the same shape, so callers in
``DMembeddingII.py`` need not change.

The first two elements of ``popt`` are synthesised so that the
downstream formula ``sigma = sqrt(2*exp(-popt[1]/popt[0]))`` recovers
the UWM σ_opt exactly. ``popt[2]`` carries the dimensionality (slope of
the linear ramp) — useful diagnostic, not used by current downstream
code. ``popt[3]`` is set to ``np.nan`` as a sentinel.

References
----------
- Original fergusonE: ``ManifoldEM/core.py``
- UWM 2026 backend:    ``ManifoldEM/uwm_ferguson_2026/`` (vendored)
- UWM user guide:      ``Running Ferguson Analysis.pdf`` (Dissanayaka, 2026)
"""
from __future__ import annotations

import warnings

import numpy as np

from .uwm_ferguson_2026 import A_ij, fit_ramp


def _estimate_N_from_D(D: np.ndarray) -> int:
    """
    Estimate the number of data points N from the length of D.

    ManifoldEM passes ``np.sqrt(yVal)`` where yVal carries the
    upper-triangle of an N×N sparse distance matrix plus structural
    zeros. The non-zero count is bounded above by N*(N-1)/2; with
    padding zeros the total length len(D) is somewhat larger.

    For the UWM ``tol = 0.05 * log(N)`` calculation, an order-of-
    magnitude estimate of N is sufficient (a 10x error in N changes
    tol by only 0.115 in absolute units, well below the curve's
    typical plateau-to-ramp gap).

    We solve N*(N-1)/2 = len(D) → N ≈ (1 + sqrt(1 + 8*len(D))) / 2,
    rounded up. This over-estimates when zeros are present, which
    pushes tol slightly higher (more conservative ramp detection).
    """
    n_est = (1.0 + np.sqrt(1.0 + 8.0 * len(D))) / 2.0
    return max(2, int(np.ceil(n_est)))


def fergusonE_uwm(
    D: np.ndarray,
    logEps: np.ndarray,
    N: int | None = None,
    a0=None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Drop-in replacement for ``ManifoldEM.core.fergusonE`` that routes
    through the UWM 2026 Ferguson backend.

    Parameters
    ----------
    D : ndarray, 1-D
        Distance vector (NOT squared distances). Same convention as
        the input to ``core.fergusonE``: callers in production pass
        ``np.sqrt(yVal)``.
    logEps : ndarray, 1-D
        Log-epsilon grid as expected by ``core.fergusonE``. The
        production grid is ``np.arange(-150, 150.2, 0.2)``. Internally
        converted to a sigma grid via ``sigma = exp(logEps/2)``.
    N : int, optional
        Number of data points, used only to set the ramp-detection
        tolerance ``tol = 0.05 * log(N)``. If ``None``, estimated from
        ``len(D)``. Pass it explicitly when known for reproducibility.
    a0 : ignored
        Kept for signature compatibility with ``core.fergusonE``. The
        UWM backend does not use an iterative initial guess.

    Returns
    -------
    popt : ndarray, shape (4,)
        Synthetic 4-element parameter vector. ``popt[0]`` and ``popt[1]``
        encode σ_opt via the production formula
        ``sigma = sqrt(2*exp(-popt[1]/popt[0]))``. ``popt[2]`` carries
        the manifold dimensionality (slope of the log-kernel-sum ramp).
        ``popt[3]`` is ``np.nan`` (sentinel).
    logSumWij : ndarray
        Log of the kernel sum at each ``logEps`` value. Same shape as
        ``logEps``.
    resnorm : float
        Sum of squared residuals of the linear fit on the central ramp.
    R_squared : float
        R² of the linear fit on the central ramp.
    """
    D = np.asarray(D, dtype=float).ravel()
    logEps = np.asarray(logEps, dtype=float).ravel()

    if N is None:
        N = _estimate_N_from_D(D)

    Dsq = D * D
    sigma = np.exp(0.5 * logEps)  # σ² ≡ ε

    # Kernel sum at each σ. The UWM A_ij accepts a vector of σ's but
    # is memory-hungry for large Dsq; the upstream analyze_.py uses a
    # for-loop for that reason. Mirror that choice here.
    A = np.empty(len(sigma), dtype=float)
    for k in range(len(sigma)):
        A[k] = A_ij(Dsq, sigma[k])[0]

    # Avoid log(0) and log of subnormal floats (mirrors UWM's
    # analyze_.py:32). With production logEps the leftmost few sigmas
    # underflow the kernel sum to 0 in float64; clipping to 1e-300
    # gives log ≈ -690, well below any ramp midpoint.
    A = np.clip(A, 1e-300, None)

    x = np.log(sigma)            # ≡ logEps / 2
    y = np.log(A)
    y = np.nan_to_num(y, neginf=-1e10)

    tol = 0.05 * np.log(N)
    p = 90.0  # central-percent of the ramp treated as linear

    xl, yl_fit, x_mid, y_mid, slope = fit_ramp(x, y, tol, p)

    sigma_opt = float(np.exp(x_mid)[0] if np.ndim(x_mid) else np.exp(x_mid))
    dimensionality = float(slope[0] if np.ndim(slope) else slope)

    # ---- Synthesise popt so downstream formula recovers σ_opt ----
    # Downstream:  sigma = sqrt(2 * exp(-popt[1] / popt[0]))
    #   ⇒  -popt[1]/popt[0] = log(sigma_opt**2 / 2)
    # Choose popt[0] = 1, then popt[1] = -log(sigma_opt**2 / 2).
    popt = np.array([
        1.0,
        -np.log(sigma_opt * sigma_opt / 2.0),
        dimensionality,
        np.nan,
    ], dtype=float)

    # logSumWij at the original logEps grid is just `y` (we already
    # computed log of kernel sum on the converted σ grid, which has
    # one-to-one correspondence with logEps).
    logSumWij = y.copy()

    # resnorm and R² from the linear fit on the central ramp.
    # fit_ramp returns yl_fit = slope*xl + intercept (the FITTED line);
    # to compute residuals we need the original y values at the same
    # indices. Recompute the linear-portion mask the same way fit_ramp
    # does, so we get matching original y values.
    y_min_arr, y_max_arr = float(np.min(y)), float(np.max(y))
    ramp_mask = np.intersect1d(
        np.where(y - y_min_arr > tol)[0],
        np.where(y_max_arr - y > tol)[0],
    )
    if len(ramp_mask) > 0:
        ramp_y0 = float(np.min(y[ramp_mask]))
        ramp_y1 = float(np.max(y[ramp_mask]))
        ramp_height = ramp_y1 - ramp_y0
        offset = 0.5 * (1.0 - 0.01 * p) * ramp_height
        linear_y0 = ramp_y0 + offset
        linear_y1 = ramp_y1 - offset
        linear_idx = np.intersect1d(
            np.where(y > linear_y0)[0],
            np.where(y < linear_y1)[0],
        )
        if len(linear_idx) >= 2:
            y_orig = y[linear_idx]
            residuals = y_orig - yl_fit
            resnorm = float(np.sum(residuals * residuals))
            ss_tot = float(np.sum((y_orig - np.mean(y_orig)) ** 2))
            R_squared = float(1.0 - resnorm / ss_tot) if ss_tot > 0 else float("nan")
        else:
            resnorm = float("nan")
            R_squared = float("nan")
    else:
        resnorm = float("nan")
        R_squared = float("nan")

    return popt, logSumWij, resnorm, R_squared
