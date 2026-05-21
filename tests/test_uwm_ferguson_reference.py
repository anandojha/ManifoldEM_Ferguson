"""
UWM 2026 Ferguson reference tests.

This module verifies the integration of the UW-Milwaukee 2026 Ferguson
backend (``ManifoldEM.uwm_ferguson_2026``) into ManifoldEM. It contains
four tests:

  1. The vendored UWM backend reproduces UWM's documented σ_opt = 0.3148.
  2. The drop-in wrapper ``fergusonE_uwm`` reproduces the same value.
  3. Cross-validation: the legacy ``core.fergusonE`` agrees with UWM
     to within a loose factor-of-3 bound.
  4. The wrapper's signature is compatible with ``core.fergusonE``.

Obtaining the test data
-----------------------
``TEST_distance_matrix.h5`` (255 MB) ships with UWM's
``Ferguson_plot_Python.zip`` archive. It is not checked into the
repository. To run these tests, place the file at one of:

  1. The path in environment variable ``UWM_FERGUSON_TEST_DATA``
  2. ``$HOME/data/uwm_ferguson_2026/TEST_distance_matrix.h5``  (default)
  3. ``tests/uwm_reference/TEST_distance_matrix.h5``           (in-repo,
     should be gitignored)

If none of these exist, all tests in this file are skipped (not failed).

Running
-------
Tests are slow on this dataset. Use ``-s`` to see the cross-validation diagnostic:

    pytest tests/test_uwm_ferguson_reference.py -v -s

The slow tests carry a ``@pytest.mark.slow`` marker; skip them in
regular runs with ``pytest -m "not slow"``.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import warnings
import pytest
import os

# Test data discovery
UWM_REFERENCE_SIGMA_OPT = 0.3148

#: Dimensionality annotation visible on the UWM ``ferguson.jpg`` plot.
UWM_REFERENCE_DIMENSIONALITY = 1.09

def _find_test_data() -> Path | None:
    """Return path to TEST_distance_matrix.h5 if found, else None."""
    candidates = []
    env_override = os.environ.get("UWM_FERGUSON_TEST_DATA")
    if env_override:
        candidates.append(Path(env_override))
    candidates.extend([
        Path.home() / "data" / "uwm_ferguson_2026" / "TEST_distance_matrix.h5",
        Path(__file__).parent / "uwm_reference" / "TEST_distance_matrix.h5",
    ])
    for p in candidates:
        if p.is_file():
            return p
    return None

TEST_DATA_PATH = _find_test_data()

requires_uwm_data = pytest.mark.skipif(
    TEST_DATA_PATH is None,
    reason=(
        "TEST_distance_matrix.h5 not found. Place it at "
        "$UWM_FERGUSON_TEST_DATA, ~/data/uwm_ferguson_2026/, "
        "or tests/uwm_reference/. See the module docstring."))

# Module fixtures: load the H5 once, share across tests
@pytest.fixture(scope="module")
def uwm_data():
    """Load yVal, yRow, N from the UWM test distance matrix."""
    if TEST_DATA_PATH is None:
        pytest.skip("UWM test data not available")
    import h5py
    with h5py.File(TEST_DATA_PATH, "r") as f:
        yVal = np.array(f["yVal"]).ravel()
        yRow = np.array(f["yRow"]).ravel()
    N = int(np.max(yRow))
    return {"yVal": yVal, "yRow": yRow, "N": N, "path": TEST_DATA_PATH}

# 1. UWM NATIVE: vendored package on UWM data must produce 0.3148
@requires_uwm_data
@pytest.mark.slow
def test_uwm_native_reproduces_known_value(uwm_data, tmp_path, monkeypatch):
    """The vendored UWM package, run on its own test file, must return
    exactly the documented σ_opt = 0.3148. This is the strongest
    regression test we have: any change to the vendored code that
    breaks this is a clear bug."""
    from ManifoldEM.uwm_ferguson_2026 import ferguson_analysis
    # Suppress the side-effect plot (run_ferguson_'s default behaviour).
    monkeypatch.setenv("DO_NOT_PLOT", "1")
    # Run from a temp dir so any output files (sigma_opt.h5, etc.) don't
    # pollute the working directory.
    monkeypatch.chdir(tmp_path)
    sigma_opt = ferguson_analysis(str(uwm_data["path"]))
    assert sigma_opt == pytest.approx(UWM_REFERENCE_SIGMA_OPT, abs=1e-4), (
        f"UWM native returned {sigma_opt}, expected "
        f"{UWM_REFERENCE_SIGMA_OPT} (within 1e-4). Vendored UWM code "
        f"may have drifted from upstream.")

# 2. WRAPPER: drop-in fergusonE_uwm on UWM data must also produce 0.3148
@requires_uwm_data
@pytest.mark.slow
def test_wrapper_reproduces_known_value(uwm_data):
    """``fergusonE_uwm`` called with UWM's data-adaptive grid (converted
    to logEps via ``logEps = 2*log(sigma)``) must reproduce σ_opt =
    0.3148 to 4 decimal places. This proves the wrapper preserves UWM
    semantics — same answer, just packaged in fergusonE's signature."""
    from ManifoldEM._ferguson_uwm import fergusonE_uwm
    from ManifoldEM.uwm_ferguson_2026 import sigma_of_interest
    yVal = uwm_data["yVal"]
    N = uwm_data["N"]
    D = np.sqrt(yVal)
    sigma_grid = sigma_of_interest(yVal)
    logEps = 2.0 * np.log(sigma_grid)
    popt, logSumWij, resnorm, R_squared = fergusonE_uwm(D, logEps, N=N)
    sigma_opt = float(np.sqrt(2.0 * np.exp(-popt[1] / popt[0])))
    # Round-to-4dp comparison mirrors run_ferguson_'s own output rounding.
    assert round(sigma_opt, 4) == UWM_REFERENCE_SIGMA_OPT, (
        f"Wrapper returned {sigma_opt:.6f} (rounds to "
        f"{round(sigma_opt, 4)}); expected {UWM_REFERENCE_SIGMA_OPT}.")
    # Dimensionality should also match the value annotated on UWM's
    # reference plot.
    assert popt[2] == pytest.approx(UWM_REFERENCE_DIMENSIONALITY, abs=0.01), (
        f"Wrapper dimensionality = {popt[2]:.4f}, expected "
        f"{UWM_REFERENCE_DIMENSIONALITY} ± 0.01.")
    # logSumWij must have the same length as logEps.
    assert logSumWij.shape == logEps.shape
    # The fit on the central ramp should be excellent — the linear-ramp
    # detection at p=90 picks the cleanest portion by construction.
    assert R_squared > 0.99, (
        f"R_squared on UWM data is {R_squared}, suspiciously low.")

# 3. CROSS-VALIDATION: current fergusonE on UWM data — informational
@requires_uwm_data
@pytest.mark.slow
def test_current_fergusonE_vs_uwm_on_same_data(uwm_data):
    """Run the current ``ManifoldEM.core.fergusonE`` on the UWM test data
    and report the σ_opt difference vs UWM's 0.3148.

    This test is **informational**, not a strict pass/fail check on
    correctness — the two backends compute σ slightly differently:

      - ``core.fergusonE`` fits a tanh and computes
        ``σ = sqrt(2 * exp(-popt[1]/popt[0]))`` (with a ``√2`` factor).
      - UWM 2026 fits a linear ramp on the central 90 % and uses
        ``σ = exp(x_mid)`` directly.

    A modest disagreement (~10–30 %) is expected from the ``√2``
    convention and the different fit window. A large disagreement
    (factor > 3) would indicate a real algorithmic divergence worth
    investigating before any backend swap.
    """
    from ManifoldEM.core import fergusonE
    yVal = uwm_data["yVal"]
    D = np.sqrt(yVal)
    logEps = np.arange(-150, 150.2, 0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, logSumWij, resnorm, R_squared = fergusonE(D, logEps)
    sigma_current = float(np.sqrt(2.0 * np.exp(-popt[1] / popt[0])))
    ratio = sigma_current / UWM_REFERENCE_SIGMA_OPT
    log_ratio = np.log(ratio)
    # Print as information - pytest -s shows this.
    print()
    print("=" * 70)
    print("CROSS-VALIDATION: current fergusonE vs UWM 2026 on UWM test data")
    print("=" * 70)
    print(f"  current core.fergusonE σ_opt: {sigma_current:.6f}")
    print(f"  UWM 2026 σ_opt:               {UWM_REFERENCE_SIGMA_OPT:.6f}")
    print(f"  ratio (current / UWM):        {ratio:.4f}")
    print(f"  log-ratio:                    {log_ratio:+.4f}")
    print(f"  current R²:                   {R_squared:.6f}")
    print("=" * 70)
    # Loose bound: factor-of-3 disagreement would be alarming. We do not
    # assert tighter than that because the methods legitimately differ.
    assert 1.0 / 3.0 < ratio < 3.0, (
        f"Current fergusonE gives σ_opt = {sigma_current:.6f}, UWM gives "
        f"{UWM_REFERENCE_SIGMA_OPT}. Ratio {ratio:.2f} is outside the "
        f"factor-of-3 bound. Investigate before considering a backend swap.")

# 4. Wrapper return shape matches fergusonE
def test_wrapper_signature_matches_fergusonE_on_synthetic_data():
    """Quick check (no UWM data needed) that the wrapper returns a 4-tuple
    of the right shapes and types, on small synthetic data. This guards
    against accidental signature drift when callers swap backends."""
    from ManifoldEM._ferguson_uwm import fergusonE_uwm
    rng = np.random.default_rng(2024)
    pts = rng.uniform(0, 1, size=(80, 3))
    diffs = pts[:, None, :] - pts[None, :, :]
    D_full = np.sqrt(np.sum(diffs ** 2, axis=-1))
    D = D_full[np.triu_indices(80, k=1)]
    D = np.concatenate([D, np.zeros(150)])  # mimic ManifoldEM padding
    logEps = np.arange(-30, 30.5, 0.5)
    popt, logSumWij, resnorm, R_squared = fergusonE_uwm(D, logEps, N=80)
    assert isinstance(popt, np.ndarray) and popt.shape == (4,)
    assert isinstance(logSumWij, np.ndarray) and logSumWij.shape == logEps.shape
    assert isinstance(resnorm, float)
    assert isinstance(R_squared, float)
    # Sanity: σ_opt back-derived from popt is positive and finite.
    sigma_opt = float(np.sqrt(2.0 * np.exp(-popt[1] / popt[0])))
    assert np.isfinite(sigma_opt) and sigma_opt > 0
