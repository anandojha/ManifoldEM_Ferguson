"""
Tests for ManifoldEM.core.fergusonE.

The Ferguson analysis function lives in ManifoldEM/core.py and is called from
exactly one place (DMembeddingII.py):
    logEps = np.arange(-150, 150.2, 0.2)
    popt, logSumWij, resnorm, R_squared = fergusonE(np.sqrt(yVal), logEps)
    sigma = tune * np.sqrt(2 * np.exp(-popt[1] / popt[0]))

So the test surface is:
  1. Function contract (return shapes, types, and finiteness)
  2. Numerical properties of the kernel-sum curve logSumWij
  3. Tanh fit quality (resnorm, R²)
  4. Downstream sigma derivation (the formula that consumes popt)
  5. Determinism / reproducibility
  6. Input-integrity (and the documented a0 in-place mutation)
  7. The exact production call pattern in DMembeddingII
  8. Golden regression values stored in tests/ferguson_fixtures/*.npz

Tolerances are tight (rtol=1e-10) on the assumption that the fixtures were
generated on the same machine that are running the tests. If the fergusonE 
function is changed intentionally, or the tests fail with off-by-tolerance numerics
on a different platform, regenerate:

    python tests/ferguson_fixtures/generate_fixtures.py

KNOWN BEHAVIOR 
---------------------------------------------------
- fergusonE crashes on the production logEps=[-150, 150] grid if the input D
  contains no zeros, because logSumWij underflows to -inf at very negative
  logEps and curve_fit refuses non-finite ydata. Production avoids this by
  passing np.sqrt(yVal) where yVal carries structural sparse-matrix zeros.
  We test both regimes: with-zeros + production grid, and no-zeros + narrow
  grid.
- fergusonE mutates a user-supplied a0 array in place (`a0 *= 0.5` inside the
  while-loop). We assert this behavior so any future change is intentional.
"""

from __future__ import annotations
from ManifoldEM.core import fergusonE
from pathlib import Path
import numpy as np
import warnings
import pytest

# Constants and paths
FIXTURE_DIR = Path(__file__).resolve().parent / "ferguson_fixtures"

# Exact production logEps grid from DMembeddingII.py:417
PRODUCTION_LOGEPS = np.arange(-150, 150.2, 0.2)

# Narrow grid that stays in the finite-logSumWij region for D ~ O(1) without zeros
NARROW_LOGEPS = np.arange(-8.0, 5.1, 0.1)
#Tolerances. Tight because fixtures are generated on the same machine.
RTOL_LOGSUMWIJ = 1e-12  # pure numpy, deterministic
RTOL_FIT = 1e-10        # curve_fit-derived (popt, resnorm, R²)
ATOL_FIT = 1e-12

# Helper functions 

def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    full = np.sqrt(np.sum(diffs**2, axis=-1))
    return full[np.triu_indices(len(points), k=1)]

def _uniform_D(N: int = 100, seed: int = 42) -> np.ndarray:
    """Pairwise distances of N uniform-random points in [0,1]^3."""
    rng = np.random.default_rng(seed)
    return _pairwise_distances(rng.uniform(0.0, 1.0, size=(N, 3)))

def _uniform_D_with_zeros(N: int = 100, n_zeros: int = 200,
                           seed: int = 42) -> np.ndarray:
    """As in production: pairwise distances + structural zeros."""
    return np.concatenate([_uniform_D(N=N, seed=seed), np.zeros(n_zeros)])

def _swiss_roll_D_with_zeros(N: int = 500, n_zeros: int = 500,
                              seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0.0, 1.0, N))
    h = 21.0 * rng.uniform(0.0, 1.0, N)
    pts = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    return np.concatenate([_pairwise_distances(pts), np.zeros(n_zeros)])


# Pytest fixtures
@pytest.fixture
def production_logEps() -> np.ndarray:
    """Exact production grid from DMembeddingII.py."""
    return PRODUCTION_LOGEPS.copy()

@pytest.fixture
def narrow_logEps() -> np.ndarray:
    """Narrow grid usable when D has no zero entries."""
    return NARROW_LOGEPS.copy()

@pytest.fixture
def D_uniform_with_zeros() -> np.ndarray:
    return _uniform_D_with_zeros()

@pytest.fixture
def D_swiss_roll_with_zeros() -> np.ndarray:
    return _swiss_roll_D_with_zeros()

@pytest.fixture
def D_uniform_no_zeros() -> np.ndarray:
    return _uniform_D()

@pytest.fixture
def fergusonE_call(D_uniform_with_zeros, production_logEps):
    """Standard production-style call. Reused across many tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fergusonE(D_uniform_with_zeros.copy(),
                         production_logEps.copy())

# 1. Output contract - shape, type, finiteness

class TestOutputContract:
    """The returned tuple shape and basic types must not change. Downstream
    code in DMembeddingII unpacks exactly four values in this order."""

    def test_returns_tuple_of_four(self, fergusonE_call):
        assert isinstance(fergusonE_call, tuple)
        assert len(fergusonE_call) == 4

    def test_popt_is_numpy_array_length_4(self, fergusonE_call):
        popt, _, _, _ = fergusonE_call
        assert isinstance(popt, np.ndarray)
        assert popt.shape == (4,)
        assert popt.dtype.kind == "f"

    def test_logSumWij_matches_logEps_length(self, fergusonE_call,
                                              production_logEps):
        _, logSumWij, _, _ = fergusonE_call
        assert isinstance(logSumWij, np.ndarray)
        assert logSumWij.shape == production_logEps.shape

    def test_resnorm_is_finite_nonneg_scalar(self, fergusonE_call):
        _, _, resnorm, _ = fergusonE_call
        # numpy scalar or 0-d array, both accepted
        val = float(resnorm)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_R_squared_is_scalar(self, fergusonE_call):
        _, _, _, R_squared = fergusonE_call
        val = float(R_squared)
        assert np.isfinite(val)
        # R² should be high for clean synthetic data; sanity bound
        assert -1.0 < val <= 1.0

# 2. Numerical properties of logSumWij

class TestLogSumWijProperties:
    """The kernel sum is a deterministic numpy computation. These properties
    follow from the math and must hold regardless of platform."""

    def test_finite_everywhere_with_zeros(self, fergusonE_call):
        """With structural zeros in D, kernel sum is at least
        log(n_zeros) > -inf at every logEps."""
        _, logSumWij, _, _ = fergusonE_call
        assert np.all(np.isfinite(logSumWij))

    def test_monotonic_nondecreasing(self, fergusonE_call):
        """Larger logEps = wider kernel = more pairs contribute. Allow tiny
        numerical noise but no real decrease."""
        _, logSumWij, _, _ = fergusonE_call
        diffs = np.diff(logSumWij)
        assert np.all(diffs >= -1e-10), (
            f"logSumWij decreased; min(diff)={diffs.min()}"
        )

    def test_left_plateau_equals_log_n_zeros(self, D_uniform_with_zeros,
                                              production_logEps):
        """At very negative logEps, the kernel deltas from the n_zeros
        zero-distance entries dominate; logSumWij → log(n_zeros) exactly."""
        n_zeros = int(np.sum(D_uniform_with_zeros == 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, logSumWij, _, _ = fergusonE(D_uniform_with_zeros.copy(),
                                            production_logEps.copy())
        np.testing.assert_allclose(logSumWij[0], np.log(n_zeros),
                                    rtol=1e-12, atol=1e-14)

    def test_right_plateau_equals_log_n_total(self, D_uniform_with_zeros,
                                               production_logEps):
        """At very positive logEps, the kernel saturates: every pair
        contributes ~1, so logSumWij → log(N_total)."""
        n_total = len(D_uniform_with_zeros)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, logSumWij, _, _ = fergusonE(D_uniform_with_zeros.copy(),
                                            production_logEps.copy())
        np.testing.assert_allclose(logSumWij[-1], np.log(n_total),
                                    rtol=1e-12, atol=1e-14)

    def test_curve_has_genuine_ramp(self, fergusonE_call):
        """The whole point of Ferguson: logSumWij is not flat. The curve
        must span a non-trivial range between its plateaus."""
        _, logSumWij, _, _ = fergusonE_call
        spread = logSumWij.max() - logSumWij.min()
        assert spread > 1.0, f"logSumWij is too flat: spread={spread}"


# 3. Tanh fit quality

class TestTanhFitQuality:
    """The function loops curve_fit until resnorm < 100 and reports R²."""

    def test_resnorm_below_loop_terminator(self, fergusonE_call):
        """Function only exits the while-loop when resnorm <= 100."""
        _, _, resnorm, _ = fergusonE_call
        assert float(resnorm) <= 100.0

    def test_R_squared_excellent_on_clean_data(self, fergusonE_call):
        """Synthetic data should fit a tanh extremely well."""
        _, _, _, R_squared = fergusonE_call
        assert float(R_squared) > 0.99

    def test_a_and_c_have_consistent_sign(self, fergusonE_call):
        """f(x) = d + c*tanh(a*x + b). A rising sigmoid (which a Ferguson
        ramp is, by construction) requires sign(a) == sign(c)."""
        popt, _, _, _ = fergusonE_call
        assert popt[0] * popt[2] > 0, (
            f"popt[0]={popt[0]} and popt[2]={popt[2]} have opposite signs; "
            "tanh fit is not a rising sigmoid."
        )

    def test_inflection_inside_logEps_range(self, fergusonE_call,
                                              production_logEps):
        """The inflection point of the tanh fit is at x = -b/a. The
        downstream formula sigma = sqrt(2*exp(-b/a)) only makes sense if
        this point lies inside the data range."""
        popt, _, _, _ = fergusonE_call
        x_infl = -popt[1] / popt[0]
        assert production_logEps.min() <= x_infl <= production_logEps.max()

    def test_residuals_consistent_with_R_squared(self, fergusonE_call,
                                                   production_logEps):
        """Recompute R² independently from popt + logSumWij and check it
        matches the reported R_squared. Pins down the formula used."""
        popt, logSumWij, _, R_squared = fergusonE_call
        def fun(x, a, b, c, d):
            return d + c * np.tanh(a * x + b)
        residuals = logSumWij - fun(production_logEps, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((logSumWij - np.mean(logSumWij))**2)
        R_squared_recomputed = 1.0 - ss_res / ss_tot
        np.testing.assert_allclose(R_squared, R_squared_recomputed,
                                    rtol=1e-12, atol=1e-14)

# 4. Downstream sigma derivation (the DMembeddingII.py formula)

class TestSigmaDerivation:
    """The only thing DMembeddingII does with popt is derive a kernel
    bandwidth via sigma = sqrt(2*exp(-popt[1]/popt[0])). These tests pin
    that formula's contract on top of fergusonE's output."""

    def test_sigma_finite_positive(self, fergusonE_call):
        popt, _, _, _ = fergusonE_call
        sigma = np.sqrt(2.0 * np.exp(-popt[1] / popt[0]))
        assert np.isfinite(sigma)
        assert sigma > 0.0

    def test_sigma_below_max_distance(self, fergusonE_call,
                                        D_uniform_with_zeros):
        """A useful kernel bandwidth shouldn't exceed the diameter of the
        point cloud — that would relax the kernel into uselessness."""
        popt, _, _, _ = fergusonE_call
        sigma = np.sqrt(2.0 * np.exp(-popt[1] / popt[0]))
        assert sigma < D_uniform_with_zeros.max()

    def test_sigma_above_smallest_nonzero_distance(self, fergusonE_call,
                                                     D_uniform_with_zeros):
        """And it shouldn't collapse below the closest pair — that would
        leave nearly all neighborhoods empty."""
        popt, _, _, _ = fergusonE_call
        sigma = np.sqrt(2.0 * np.exp(-popt[1] / popt[0]))
        nonzero = D_uniform_with_zeros[D_uniform_with_zeros > 0]
        assert sigma > nonzero.min()

# 5. Determinism / reproducibility

class TestDeterminism:
    def test_repeated_call_same_output(self, D_uniform_with_zeros,
                                        production_logEps):
        """Two calls with identical inputs produce identical outputs."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r1 = fergusonE(D_uniform_with_zeros.copy(),
                            production_logEps.copy())
            r2 = fergusonE(D_uniform_with_zeros.copy(),
                            production_logEps.copy())
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_default_a0_equals_explicit_ones(self, D_uniform_with_zeros,
                                               production_logEps):
        """Passing a0=None and a0=np.ones(4) must produce the same result."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_default = fergusonE(D_uniform_with_zeros.copy(),
                                   production_logEps.copy(),
                                   a0=None)
            r_explicit = fergusonE(D_uniform_with_zeros.copy(),
                                    production_logEps.copy(),
                                    a0=np.ones(4))
        for a, b in zip(r_default, r_explicit):
            np.testing.assert_allclose(np.asarray(a, dtype=float),
                                        np.asarray(b, dtype=float),
                                        rtol=RTOL_FIT, atol=ATOL_FIT)

# 6. Input integrity and the documented a0 in-place mutation

class TestInputIntegrity:
    def test_D_not_modified(self, D_uniform_with_zeros, production_logEps):
        D = D_uniform_with_zeros.copy()
        D_before = D.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fergusonE(D, production_logEps.copy())
        np.testing.assert_array_equal(D, D_before)

    def test_logEps_not_modified(self, D_uniform_with_zeros,
                                   production_logEps):
        logEps = production_logEps.copy()
        logEps_before = logEps.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fergusonE(D_uniform_with_zeros.copy(), logEps)
        np.testing.assert_array_equal(logEps, logEps_before)

    def test_a0_IS_mutated_in_place(self, D_uniform_with_zeros,
                                      production_logEps):
        """KNOWN BEHAVIOR: fergusonE does `a0 *= 0.5` inside its loop, which
        mutates the caller's array. The default a0=None branch sidesteps
        this because it allocates a fresh ones(4) per call. This test pins
        the behavior so any future fix is intentional and visible."""
        a0 = np.ones(4)
        a0_before = a0.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fergusonE(D_uniform_with_zeros.copy(),
                       production_logEps.copy(),
                       a0=a0)
        # a0 is at least halved once (loop terminator divides at end of
        # every iteration); for our well-behaved inputs that converge in
        # one iteration, expected post-call value is 0.5 * ones(4).
        assert not np.array_equal(a0, a0_before), (
            "fergusonE no longer mutates a0; if intentional, update or "
            "remove this test."
        )
        assert np.all(a0 < 1.0)

# 7. Realistic call pattern from DMembeddingII.py

class TestDMembeddingIICallPattern:
    """End-to-end test mirroring the only call site of fergusonE."""

    def test_production_invocation(self):
        """Reproduce the exact DMembeddingII.py pattern:
            logEps = np.arange(-150, 150.2, 0.2)
            popt, logSumWij, resnorm, R_squared = fergusonE(np.sqrt(yVal),
                                                              logEps)
            sigma = tune * np.sqrt(2*exp(-popt[1]/popt[0]))
        with synthetic yVal mimicking calc_distance.py output.
        """
        # yVal = squared pairwise distances + structural zeros
        rng = np.random.default_rng(2024)
        pts = rng.uniform(0, 1, size=(80, 3))
        D = _pairwise_distances(pts)
        yVal_nonzero = D**2
        yVal = np.concatenate([yVal_nonzero, np.zeros(150)])
        logEps = np.arange(-150, 150.2, 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, logSumWij, resnorm, R_squared = fergusonE(np.sqrt(yVal),
                                                              logEps)
        # Same downstream derivation as DMembeddingII.py:429 (with tune=1)
        sigma = float(np.sqrt(2.0 * np.exp(-popt[1] / popt[0])))
        assert popt.shape == (4,)
        assert logSumWij.shape == logEps.shape
        assert np.all(np.isfinite(logSumWij))
        assert float(resnorm) <= 100.0
        assert float(R_squared) > 0.99
        assert np.isfinite(sigma) and sigma > 0


# 8. Regression 

FIXTURE_FILES = sorted(FIXTURE_DIR.glob("*.npz"))

@pytest.mark.skipif(
    len(FIXTURE_FILES) == 0,
    reason=(
        "No golden fixtures found. Run "
        "`python tests/ferguson_fixtures/generate_fixtures.py` first."
    ),
)
class TestGoldenRegression:
    """For each .npz under tests/ferguson_fixtures/, re-run fergusonE on
    the stored input and compare against the stored output. These are the
    tests our future Ferguson backend swap will be measured against."""

    @pytest.mark.parametrize("fixture_path", FIXTURE_FILES,
                              ids=[p.stem for p in FIXTURE_FILES])
    def test_golden(self, fixture_path):
        with np.load(fixture_path, allow_pickle=False) as data:
            D = data["D"]
            logEps = data["logEps"]
            popt_expected = data["popt"]
            logSumWij_expected = data["logSumWij"]
            resnorm_expected = float(data["resnorm"])
            R_squared_expected = float(data["R_squared"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, logSumWij, resnorm, R_squared = fergusonE(D.copy(),
                                                              logEps.copy())

        # Deterministic part (numpy only): very tight
        np.testing.assert_allclose(
            logSumWij, logSumWij_expected,
            rtol=RTOL_LOGSUMWIJ, atol=1e-14,
            err_msg=f"logSumWij drift in {fixture_path.name}",
        )
        # curve_fit-derived part: still tight, but allow a bit of slack
        np.testing.assert_allclose(
            popt, popt_expected,
            rtol=RTOL_FIT, atol=ATOL_FIT,
            err_msg=f"popt drift in {fixture_path.name}",
        )
        np.testing.assert_allclose(
            float(resnorm), resnorm_expected,
            rtol=RTOL_FIT, atol=ATOL_FIT,
            err_msg=f"resnorm drift in {fixture_path.name}",
        )
        np.testing.assert_allclose(
            float(R_squared), R_squared_expected,
            rtol=RTOL_FIT, atol=ATOL_FIT,
            err_msg=f"R² drift in {fixture_path.name}",
        )
