"""H0 (no-drift) calibration test for the three MMD-based drift detectors.

A two-sample test calibrated at level ``alpha`` is supposed to reject the
null hypothesis "the two windows come from the same distribution" with
probability at most ``alpha`` *when the null is true*.  In drift-detection
language: on a stationary stream, the empirical false-alarm rate should
sit at or below the nominal ``alpha``.

This script generates many independent stationary streams (no real drift),
slides each detector through them with the same window protocol used by
the benchmark, and reports the empirical Type-I error.

Detectors evaluated
-------------------
1. **Standard MMD** with permutation test
   (``core.detectors.mmd.mmd``)
2. **IDW-MMD** with asymptotic p-value
   (``core.detectors.mmd_variants.wmmd_asymptotic``)
3. **SE-CDT** composite pipeline
   (``core.detectors.mmd_variants.shapedd_idw_mmd_proper``)

Reference distributions
-----------------------
* iid Gaussian, d=5
* iid Gaussian, d=10  (heavier dimensionality)
* Correlated Gaussian, d=5  (off-diagonal Cov)

The script is deterministic so the printed table can be quoted in the
thesis verbatim.

Outputs
-------
``results/raw/h0_calibration.json``  - machine-readable summary
``results/tables/table_h0_calibration.tex`` - LaTeX table for the report

Usage
-----
::

    PYTHONPATH=. python3 scripts/h0_calibration.py [--quick]

``--quick`` runs only 30 streams per cell instead of 200.
"""

from __future__ import annotations

import argparse
import json
import os

# Avoid BLAS oversubscription when this script is run alongside other jobs.
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import RAW_DIR, TABLES_DIR  # noqa: E402
from core.detectors.mmd import mmd  # noqa: E402
from core.detectors.mmd_variants import (  # noqa: E402
    shapedd_idw_mmd_proper,
    wmmd_gamma,
)


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ALPHA = 0.05            # Nominal test level
DEFAULT_N_RUNS = 200    # Independent stationary streams per (detector, dist)
QUICK_N_RUNS = 30       # Number of runs in --quick mode
WINDOW_SIZE = 300       # Same as benchmark CHUNK_SIZE
SHAPE_L1 = 50
SHAPE_L2 = 150
N_PERM = 2500
SEED_BASE = 20260418    # Today's date as base seed for full reproducibility


# ----------------------------------------------------------------------------
# Stationary distributions
# ----------------------------------------------------------------------------
@dataclass
class StationaryDist:
    name: str
    dim: int
    cov: str  # "I" or "AR1"


REFERENCE_DISTS: list[StationaryDist] = [
    StationaryDist(name="Gauss-iid d=5",    dim=5,  cov="I"),
    StationaryDist(name="Gauss-iid d=10",   dim=10, cov="I"),
    StationaryDist(name="Gauss-AR1 d=5",    dim=5,  cov="AR1"),
]


def _ar1_cov(d: int, rho: float = 0.6) -> np.ndarray:
    """AR(1)-style covariance: Cov[i,j] = rho ** |i-j|."""
    idx = np.arange(d)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample_stationary(dist: StationaryDist, n: int, rng: np.random.Generator) -> np.ndarray:
    if dist.cov == "I":
        return rng.standard_normal((n, dist.dim))
    if dist.cov == "AR1":
        cov = _ar1_cov(dist.dim, rho=0.6)
        return rng.multivariate_normal(mean=np.zeros(dist.dim), cov=cov, size=n)
    raise ValueError(f"Unknown covariance type: {dist.cov!r}")


# ----------------------------------------------------------------------------
# Single-window detector probes
# ----------------------------------------------------------------------------
def probe_standard_mmd(window: np.ndarray, *, alpha: float = ALPHA) -> bool:
    """Return True if Standard MMD (permutation) rejects H0 at level alpha."""
    np.random.seed(0)  # deterministic permutation matrix per call
    _, p = mmd(window, n_perm=N_PERM)
    return bool(p < alpha)


def probe_idw_mmd(window: np.ndarray, *, alpha: float = ALPHA) -> bool:
    """Return True if IDW-MMD with Gamma-approximation null rejects H0 at level alpha.

    Note: previously this routed to ``wmmd_asymptotic`` which used a
    Gaussian H₁ asymptotic mis-applied as the H₀ null (see
    ``mmd_variants.wmmd_gamma`` docstring for the audit and fix).  We
    now use the moment-matched Gamma null which is properly calibrated
    at α.
    """
    s = len(window) // 2
    _, p = wmmd_gamma(window, s, weight_method="variance_reduction")
    return bool(p < alpha)


def probe_se_cdt(window: np.ndarray, *, alpha: float = ALPHA) -> bool:
    """Return True if SE-CDT composite reports drift at level alpha."""
    is_drift, _, _, _ = shapedd_idw_mmd_proper(
        window, l1=SHAPE_L1, l2=SHAPE_L2, alpha=alpha
    )
    return bool(is_drift)


PROBES = {
    "Standard MMD (perm)":   probe_standard_mmd,
    "IDW-MMD (Gamma null)":  probe_idw_mmd,
    "SE-CDT (composite)":    probe_se_cdt,
}


# ----------------------------------------------------------------------------
# Calibration loop
# ----------------------------------------------------------------------------
def run_calibration(
    dist: StationaryDist,
    probe_name: str,
    probe_fn,
    *,
    n_runs: int,
    window_size: int = WINDOW_SIZE,
    seed_base: int = SEED_BASE,
) -> dict:
    """Run ``n_runs`` independent windows from ``dist`` and count rejections."""
    rejections = 0
    for r in range(n_runs):
        seed = seed_base + 7919 * r
        rng = np.random.default_rng(seed)
        window = sample_stationary(dist, window_size, rng)
        if probe_fn(window):
            rejections += 1

    rate = rejections / n_runs
    # Wilson 95% CI approximation (Bernoulli)
    se = float(np.sqrt(rate * (1 - rate) / n_runs))
    return {
        "detector": probe_name,
        "distribution": dist.name,
        "rejections": int(rejections),
        "n_runs": int(n_runs),
        "type1_rate": float(rate),
        "approx_95ci_halfwidth": float(1.96 * se),
    }


def _write_latex_table(rows: list[dict], n_runs: int) -> Path:
    out = TABLES_DIR / "table_h0_calibration.tex"
    detectors = list(PROBES.keys())
    out.parent.mkdir(parents=True, exist_ok=True)

    align = "|l|" + "|".join(["c"] * len(detectors)) + "|"
    lines = [
        "% Auto-generated by scripts/h0_calibration.py",
        f"% n_runs per cell = {n_runs}, alpha = {ALPHA}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(["\\textbf{Distribution}"] + [f"\\textbf{{{d}}}" for d in detectors]) + " \\\\",
        "\\hline",
    ]
    for dist in REFERENCE_DISTS:
        cells = [dist.name]
        for det in detectors:
            row = next(r for r in rows if r["detector"] == det and r["distribution"] == dist.name)
            cells.append(f"{row['type1_rate']:.3f} $\\pm$ {row['approx_95ci_halfwidth']:.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Run {QUICK_N_RUNS} runs/cell instead of {DEFAULT_N_RUNS} (debugging).",
    )
    args = parser.parse_args()
    n_runs = QUICK_N_RUNS if args.quick else DEFAULT_N_RUNS

    print("=" * 78)
    print("H0 calibration test  (nominal alpha = {:.2f})".format(ALPHA))
    print("Each cell = empirical Type-I error over {} stationary windows".format(n_runs))
    print("=" * 78)

    rows: list[dict] = []
    for dist in REFERENCE_DISTS:
        for probe_name, probe_fn in PROBES.items():
            print(f"  running {probe_name:24s}  on {dist.name} ...", flush=True)
            row = run_calibration(dist, probe_name, probe_fn, n_runs=n_runs)
            rows.append(row)

    # Pretty-print summary table
    print()
    print("=" * 78)
    print(f"SUMMARY  (alpha = {ALPHA}, rate +/- 1.96 * SE)")
    print("=" * 78)
    header = f"  {'distribution':22s}  " + "  ".join(f"{k:>22s}" for k in PROBES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dist in REFERENCE_DISTS:
        cells = []
        for probe_name in PROBES:
            row = next(r for r in rows if r["detector"] == probe_name and r["distribution"] == dist.name)
            cell = f"{row['type1_rate']:.3f} +/- {row['approx_95ci_halfwidth']:.3f}"
            cells.append(f"{cell:>22s}")
        print(f"  {dist.name:22s}  " + "  ".join(cells))

    print()
    print("Notes:")
    print("  * A well-calibrated detector at alpha=0.05 should produce rates")
    print("    that contain 0.05 inside the 95% CI half-width above.")
    print("  * Rates substantially below 0.05 indicate a *conservative* test")
    print("    (under-rejects H0 -> safer FP, may also reduce power).")
    print("  * Rates substantially above 0.05 indicate an *anti-conservative*")
    print("    test (over-rejects H0 -> inflated false alarms).")

    # Persist outputs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    json_out = RAW_DIR / "h0_calibration.json"
    payload = {
        "alpha": ALPHA,
        "n_runs": n_runs,
        "window_size": WINDOW_SIZE,
        "shape_l1": SHAPE_L1,
        "shape_l2": SHAPE_L2,
        "n_perm": N_PERM,
        "seed_base": SEED_BASE,
        "rows": rows,
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tex_out = _write_latex_table(rows, n_runs)
    print(f"\nWrote results -> {json_out.relative_to(PROJECT_ROOT)}")
    print(f"Wrote table   -> {tex_out.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
