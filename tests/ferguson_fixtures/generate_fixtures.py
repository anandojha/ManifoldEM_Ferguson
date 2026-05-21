"""
Generate regression fixtures for ManifoldEM.core.fergusonE.

fergusonE wraps scipy.optimize.curve_fit (Levenberg-Marquardt) whose iterative
output is sensitive to BLAS/LAPACK implementation. Bit-identical reproducibility
across platforms (e.g., Linux x86_64 vs macOS arm64) is not guaranteed, so we
generate the golden values on the SAME machine that runs the tests.

USAGE
-----
Run once after install (and any time the fergusonE function is changed):
    cd /path/to/ManifoldEM
    python tests/ferguson_fixtures/generate_fixtures.py

This writes one .npz file per fixture into this directory. Tests load them
back and compare with tight tolerances.

EACH FIXTURE
------------
A .npz file with these arrays:
    D            : 1D distance vector (input)
    logEps       : 1D log-epsilon grid (input)
    popt         : 4-element tanh-fit parameters (output)
    logSumWij    : log of kernel sum at each logEps (output)
    resnorm      : scalar (output, stored as 0-d array)
    R_squared    : scalar (output, stored as 0-d array)
    a0_input     : a0 array passed in (or default ones(4))
    description  : 0-d string explaining what the input represents
"""
from __future__ import annotations
from ManifoldEM.core import fergusonE
from pathlib import Path
import numpy as np
import warnings

# Synthetic data generators (deterministic, seeded)

def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Upper-triangular (k=1) Euclidean pairwise distances of N points in R^d."""
    diffs = points[:, None, :] - points[None, :, :]
    full = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(len(points), k=1)
    return full[iu]

def gen_uniform_3d_with_zeros(N: int = 100, n_zeros: int = 200,
                               seed: int = 42) -> np.ndarray:
    """
    Pairwise distances of N uniform-random points in [0,1]^3, plus n_zeros
    explicit zero entries.

    Mimics the production input pattern: in DMembeddingII.py the sparse
    distance matrix carries structural zeros that propagate into
    np.sqrt(yVal). Without them, the kernel sum in fergusonE underflows to
    -inf at the left edge of the production logEps grid and curve_fit fails.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(N, 3))
    D = _pairwise_distances(pts)
    return np.concatenate([D, np.zeros(n_zeros)])

def gen_swiss_roll_with_zeros(N: int = 500, n_zeros: int = 500,
                               seed: int = 7) -> np.ndarray:
    """Swiss roll: 2D manifold in R^3, with structural zeros."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0.0, 1.0, N))
    h = 21.0 * rng.uniform(0.0, 1.0, N)
    pts = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    D = _pairwise_distances(pts)
    return np.concatenate([D, np.zeros(n_zeros)])

def gen_sphere_with_zeros(N: int = 300, n_zeros: int = 300,
                          seed: int = 11) -> np.ndarray:
    """N points on S^2 (chordal distances), with structural zeros."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 1.0, size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    D = _pairwise_distances(pts)
    return np.concatenate([D, np.zeros(n_zeros)])

def gen_uniform_3d_no_zeros(N: int = 100, seed: int = 42) -> np.ndarray:
    """Same as uniform_3d but no zeros — only safe with a narrow logEps grid."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(N, 3))
    return _pairwise_distances(pts)

# logEps grids
# Exact production grid from DMembeddingII.py
PRODUCTION_LOGEPS = np.arange(-150, 150.2, 0.2)
# Narrow grid that stays in the finite region for D ~ O(1) without zeros
NARROW_LOGEPS = np.arange(-8.0, 5.1, 0.1)

# Fixture catalog

FIXTURES = [
    {
        "name": "uniform_3d_with_zeros_production_grid",
        "D": gen_uniform_3d_with_zeros(),
        "logEps": PRODUCTION_LOGEPS,
        "description": (
            "100 uniform points in [0,1]^3 + 200 zero entries; production "
            "logEps grid (-150 to 150 step 0.2). Mirrors DMembeddingII call."
        ),
    },
    {
        "name": "swiss_roll_with_zeros_production_grid",
        "D": gen_swiss_roll_with_zeros(),
        "logEps": PRODUCTION_LOGEPS,
        "description": (
            "500 points on a Swiss-roll 2-manifold + 500 zero entries; "
            "production logEps grid."
        ),
    },
    {
        "name": "sphere_with_zeros_production_grid",
        "D": gen_sphere_with_zeros(),
        "logEps": PRODUCTION_LOGEPS,
        "description": (
            "300 points on the unit sphere (chordal distances) + 300 zero "
            "entries; production logEps grid."
        ),
    },
    {
        "name": "uniform_3d_no_zeros_narrow_grid",
        "D": gen_uniform_3d_no_zeros(),
        "logEps": NARROW_LOGEPS,
        "description": (
            "100 uniform points in [0,1]^3, no zeros; narrow logEps grid "
            "(-8 to 5 step 0.1) that stays in the finite-logSumWij region."
        ),
    },
]

# Driver

def main() -> None:
    out_dir = Path(__file__).resolve().parent
    print(f"Writing fixtures to: {out_dir}")
    print("-" * 70)

    for spec in FIXTURES:
        name = spec["name"]
        D = spec["D"].copy()             # copy: fergusonE should not mutate, but be safe
        logEps = spec["logEps"].copy()
        a0_input = np.ones(4)            # explicit, so we capture the post-call mutation
        a0_for_call = a0_input.copy()    # the function mutates this, we store the input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, logSumWij, resnorm, R_squared = fergusonE(
                D, logEps, a0=a0_for_call.copy()
            )
        out_path = out_dir / f"{name}.npz"
        np.savez(
            out_path,
            D=D,
            logEps=logEps,
            popt=np.asarray(popt),
            logSumWij=np.asarray(logSumWij),
            resnorm=np.asarray(resnorm),
            R_squared=np.asarray(R_squared),
            a0_input=a0_input,
            description=np.array(spec["description"]),
        )
        sigma = float(np.sqrt(2.0 * np.exp(-popt[1] / popt[0])))
        print(f"  {name}")
        print(f"    D shape={D.shape}, n_zeros={int(np.sum(D == 0))}")
        print(f"    popt={popt}")
        print(f"    R²={R_squared:.6f}   resnorm={resnorm:.6f}   "
              f"sigma={sigma:.6f}")
        print(f"    -> {out_path.name}")
    print("-" * 70)
    print("Done.")

if __name__ == "__main__":
    main()
