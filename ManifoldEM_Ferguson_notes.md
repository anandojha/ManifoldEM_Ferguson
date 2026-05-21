####################################################################################
1. conda activate manifoldem
2. /mnt/home/aojha/ManifoldEM_Ferguson  --> uwm_ferguson_2026
3. pytest tests/test_ferguson.py --collect-only -q | tail -5
4. pytest tests/test_uwm_ferguson_reference.py --collect-only -q | tail -5
5. cd uwm_reference/ --> TEST_distance_matrix.h5
Three of the four tests in test_uwm_ferguson_reference.py load this file. The fourth uses synthetic data we generate on the fly.
6. cd ferguson_fixtures; python generate_fixtures.py
Fixtures regenerated locally, i.e., 4 geometries (3D cube, Swiss roll, sphere, narrow-grid edge case), σ values span 0.05-5.6.
7. pytest test_ferguson.py
8. pytest test_uwm_ferguson_reference.py
9. pytest test_uwm_ferguson_reference.py::test_current_fergusonE_vs_uwm_on_same_data -v -s
10. echo "MANIFOLDEM_FERGUSON_BACKEND=$MANIFOLDEM_FERGUSON_BACKEND"

####################################################################################
# ManifoldEM Ferguson Integration — Notes

## 1. Ferguson analysis

Ferguson analysis is a method for **automatically picking the bandwidth (σ)** of a Gaussian kernel directly from the data, by examining how the sum of pairwise kernel values changes with σ. It produces a single number, **σ_opt**, which is then plugged into the diffusion-map kernel `exp(−D²/(2σ²))` that drives the rest of ManifoldEM's embedding pipeline.

As σ sweeps from very small to very large values, the log of the kernel sum traces out a sigmoidal curve which is flat at small σ (no neighbors connect, kernel sum ≈ number of self-pairs), flat at large σ (every pair is fully connected, kernel sum ≈ N²/2), with a ramp in between. The midpoint of that ramp is σ_opt, i.e., the bandwidth that best balances locality and connectivity for the intrinsic geometry of the data.

**Where it happens in ManifoldEM.** Once per connected component, inside `DMembeddingII.py` (line 418). The resulting σ_opt feeds the kernel that produces the diffusion-map eigenvectors, which become the conformational coordinates downstream. In the CLI pipeline, this corresponds to the `manifold-analysis` step.

### The two backends

This integration makes available **two different implementations** of Ferguson analysis:

| Backend | How it picks σ_opt | Bonus output |
|---------|-------------------|--------------|
| **Legacy** (`ManifoldEM.core.fergusonE`) | Fits a 4-parameter tanh sigmoid to the log-kernel-sum curve and σ_opt comes from the inflection point | — |
| **UWM 2026** (`ManifoldEM.uwm_ferguson_2026`) | Fits a linear regression to the central 90 % of the ramp and σ_opt comes from the midpoint of that linear segment | manifold **dimensionality** (slope of the ramp) |

Both produce σ_opt from the same input (pairwise distances). The two methods are mathematically distinct but yield σ_opt values that **agree to 1.16 % on UWM's reference dataset** (see the cross-validation test below).

The integration adds the UWM backend as an opt-in alternative without changing the default. Switching between them is one environment variable.

---

## 2. Tests

The integration ships with **40 tests** in two new files:

| File | Test count | What it covers |
|------|------------|----------------|
| `tests/test_ferguson.py` | 28 | Regression coverage of the **legacy** `core.fergusonE` |
| `tests/test_uwm_ferguson_reference.py` | 4 | Reference tests for the **UWM 2026** backend and its wrapper |

(Plus 8 pre-existing tests in `test_core.py` and `test_quaternion.py`, unrelated to Ferguson.)

### What `tests/test_ferguson.py` (28 tests) verifies

These exist because legacy `fergusonE` had **zero test coverage** before this work. They define exactly what its behavioral contract is, so any future change is a deliberate decision rather than an accidental break.

| Test class | # | What it pins down | Why it matters |
|-----------|---|-------------------|----------------|
| `TestOutputContract` | 5 | Returns a 4-tuple; `popt` is shape-(4,); `logSumWij` length matches `logEps`; `resnorm` and `R²` are scalars | If the return shape ever changes, downstream code in `DMembeddingII` breaks silently |
| `TestLogSumWijProperties` | 5 | Log-kernel-sum curve is monotonic, finite, plateaus at `log(n_zeros)` and `log(N_total)`, has a non-trivial ramp | These mathematical facts must always hold; if they don't, the kernel sum computation is broken |
| `TestTanhFitQuality` | 5 | `resnorm ≤ 100` (the loop terminator); R² > 0.99 on clean data; tanh inflection inside `logEps` range; R² recomputable from `popt` | Verifies the curve fit is actually converging correctly |
| `TestSigmaDerivation` | 3 | Recovered σ is finite, positive, between the smallest non-zero pair distance and the diameter of the point cloud | Sanity bounds on the actual quantity used downstream in `DMembeddingII` |
| `TestDeterminism` | 2 | Repeated calls give identical output; `a0=None` matches `a0=np.ones(4)` | Reproducibility guarantee |
| `TestInputIntegrity` | 3 | `D` and `logEps` are not modified; **`a0` is mutated in-place** | Pins both the safe behaviors and the known in-place mutation, so any future fix is intentional rather than accidental |
| `TestDMembeddingIICallPattern` | 1 | End-to-end call mirroring `DMembeddingII.py:418` | Catches breakage at the actual production call site, not just an idealized one |
| `TestGoldenRegression` | 4 | Bit-tight comparison against pre-computed fixtures: uniform 3D cube, Swiss roll, sphere, no-zeros narrow grid | Catches sub-1e-10 numerical drift; fixtures are platform-specific and regenerated per machine |

### What `tests/test_uwm_ferguson_reference.py` (4 tests) verifies

These verify the UWM 2026 integration is correct and quantify how its results compare to legacy.

| Test | What it verifies | Why it matters |
|------|------------------|----------------|
| `test_uwm_native_reproduces_known_value` | UWM code on UWM's own test data produces σ_opt = 0.3148 | Strongest regression test possible that reproduces the value the upstream package documents in `sigma_opt.h5` |
| `test_wrapper_reproduces_known_value` | The drop-in `fergusonE_uwm` wrapper also produces σ_opt = 0.3148 and dimensionality = 1.09 | Proves the wrapper preserves UWM semantics - almost same answer, packaged in the legacy `fergusonE` |
| `test_current_fergusonE_vs_uwm_on_same_data` | Reports the σ_opt difference between legacy and UWM on the same dataset (factor-of-3 sanity bound) | This is the scientific result that quantifies how much σ_opt moves between backends. Result: 1.16 % on UWM's reference data |
| `test_wrapper_signature_matches_fergusonE_on_synthetic_data` | Wrapper returns a 4-tuple with the right shapes and types | Guards against signature drift if the wrapper is ever modified |

### Running the tests

```bash
cd ~/code/ManifoldEM
pytest tests/ -v
```

Expected: **40 passed in ~4-5 minutes**. The 4 UWM tests need `TEST_distance_matrix.h5` to be findable. See the docstring in `tests/test_uwm_ferguson_reference.py` for the setup. If the data file is not present, those 4 tests are **skipped** (not failed) and we would see "36 passed, 4 skipped".

To see the cross-validation diagnostic:

```bash
pytest tests/test_uwm_ferguson_reference.py::test_current_fergusonE_vs_uwm_on_same_data -v -s
```

Output:

```
======================================================================
CROSS-VALIDATION: current fergusonE vs UWM 2026 on UWM test data
======================================================================
  current core.fergusonE σ_opt: 0.318440
  UWM 2026 σ_opt:               0.314800
  ratio (current / UWM):        1.0116
  log-ratio:                    +0.0115
======================================================================
```

---

## 3. Using the new backend in the ManifoldEM CLI

### 3.1 Where Ferguson runs in the pipeline

ManifoldEM is invoked via the `manifold-cli` command (defined in `pyproject.toml` as `[project.scripts] manifold-cli = "ManifoldEM.interfaces.cli:main"`). The full pipeline runs as nine sub-steps:

```bash
manifold-cli init                  # 0 — initialize project
manifold-cli threshold             # 1 — threshold setting
manifold-cli calc-distance         # 2 — pairwise S² distances
manifold-cli manifold-analysis     # 3 — Intial embedding (Ferguson runs here)
manifold-cli psi-analysis          # 4
manifold-cli nlsa-movie            # 5
manifold-cli find-ccs              # 7
manifold-cli calc-probabilities    # 8
manifold-cli trajectory            # 9
```

**Ferguson runs only in step 3 (`manifold-analysis`).** The selected σ_opt is then saved to the project state file, and downstream steps use that stored value as they do not recompute it. So `MANIFOLDEM_FERGUSON_BACKEND` only matters when `manifold-analysis` is being run.

### 3.2 Default behavior (unchanged)

```bash
manifold-cli -n 16 manifold-analysis params_my_analysis.toml
```

Uses `ManifoldEM.core.fergusonE` (legacy). **No change vs. before this integration.**

### 3.3 Switch to UWM 2026

Set the environment variable `MANIFOLDEM_FERGUSON_BACKEND=uwm` before running `manifold-analysis`. There are three ways to run it.

**(1) Inline, for one command only:**

```bash
MANIFOLDEM_FERGUSON_BACKEND=uwm manifold-cli -n 16 manifold-analysis params_my_analysis.toml
```

**(2) For a whole shell session:**

```bash
export MANIFOLDEM_FERGUSON_BACKEND=uwm
manifold-cli -n 16 manifold-analysis params_my_analysis.toml       # uses UWM
unset MANIFOLDEM_FERGUSON_BACKEND                                  # back to legacy
```

**(3) Inside a SLURM job script (typical Rusty / ccblin091 workflow):**

```bash
#!/bin/bash
#SBATCH --job-name=manifoldem-uwm
#SBATCH --output=manifoldem_%j.out
#SBATCH --error=manifoldem_%j.err
#SBATCH --partition=ccb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128000
#SBATCH --time=24:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate manifoldem

# Select Ferguson backend
export MANIFOLDEM_FERGUSON_BACKEND=uwm

# Sanity check - Log which backend will run
python -c "from ManifoldEM import DMembeddingII; print('Ferguson backend:', DMembeddingII.fergusonE.__module__)"

# Pipeline
cd /path/to/your/project
PARAMS=params_my_analysis.toml
NCPU=$SLURM_CPUS_PER_TASK

manifold-cli -n $NCPU calc-distance        $PARAMS
manifold-cli -n $NCPU manifold-analysis    $PARAMS    # Ferguson runs here
manifold-cli -n $NCPU psi-analysis         $PARAMS
manifold-cli -n $NCPU nlsa-movie           $PARAMS
manifold-cli -n $NCPU find-ccs             $PARAMS
manifold-cli -n $NCPU calc-probabilities   $PARAMS
manifold-cli -n $NCPU trajectory           $PARAMS
```

The SLURM `.out` log will show which backend was selected at the top:

```
Ferguson backend: ManifoldEM._ferguson_uwm
ManifoldEM version: 0.3.1...
```

If we forgot the `export` line, the sanity check shows `ManifoldEM.core` and we immediately know the legacy backend was used.

### 3.4 Verify which backend is active

Before launching a long job, confirm which backend will be used:

```bash
python -c "from ManifoldEM import DMembeddingII; print(DMembeddingII.fergusonE.__module__)"
```

| Output | Backend |
|--------|---------|
| `ManifoldEM.core` | Legacy |
| `ManifoldEM._ferguson_uwm` | UWM 2026 |

### 3.5 Re-running just the embedding step

If we have already done a full pipeline with legacy and want to compare against UWM **without redoing the slow `calc-distance` step**, just re-run from `manifold-analysis` onward:

```bash
export MANIFOLDEM_FERGUSON_BACKEND=uwm
manifold-cli -n 16 manifold-analysis params_my_analysis.toml
manifold-cli -n 16 psi-analysis      params_my_analysis.toml
# ... and any downstream steps you want to compare
```

`calc-distance` produces the S² geodesic distance matrix and that input is independent of the Ferguson backend, so it never needs to be re-run.

### 3.6 A/B testing two backends on the same dataset

Run both pipelines in parallel by sharing the expensive `calc-distance` output across two project files:

```bash
# Initial setup (run once)
manifold-cli init -p compare_legacy ...
manifold-cli threshold ...           params_compare_legacy.toml
manifold-cli -n 16 calc-distance     params_compare_legacy.toml

# Make a UWM copy that shares the same distance results
cp params_compare_legacy.toml params_compare_uwm.toml
# (manually edit project_name inside, or copy the project directory)

# Run legacy embedding
manifold-cli -n 16 manifold-analysis params_compare_legacy.toml

# Run UWM embedding
export MANIFOLDEM_FERGUSON_BACKEND=uwm
manifold-cli -n 16 manifold-analysis params_compare_uwm.toml

# Compare σ_opt and downstream eigenvectors between the two project outputs
```
