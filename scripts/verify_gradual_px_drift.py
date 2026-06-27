"""Verify whether each gradual benchmark dataset induces a genuine P(X) drift.

Unsupervised drift detectors only see the marginal ``P(X)``.  A dataset where
only ``P(Y|X)`` changes (e.g. only the labelling rule shifts) is invisible to
them and should not be used to evaluate gradual P(X) detection.

This script generates each candidate gradual dataset twice (once for the
pre-drift segment, once for the post-drift segment) and runs a per-feature
two-sample Kolmogorov-Smirnov test.  A dataset is flagged as having a
genuine P(X) drift if any feature passes the standard 5%-significance test
with a Bonferroni correction across the ``d`` features.

Usage::

    python -m scripts.verify_gradual_px_drift

The script is deterministic (fixed seed) so its output can be quoted in
the thesis verbatim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generators.benchmark_generators import generate_drift_stream  # noqa: E402

CANDIDATE_DATASETS = [
    {
        "name": "circles_gradual",
        "config": {
            "type": "circles_gradual",
            "n_drift_events": 10,
            "params": {"transition_width": 400},
        },
    },
    {
        "name": "gaussian_gradual_synthetic",
        "config": {
            "type": "gaussian_gradual",
            "n_drift_events": 10,
            "params": {
                "n_features": 10,
                "shift_magnitude": 1.5,
                "transition_width": 400,
            },
        },
    },
]

TOTAL_SIZE = 10_000
SEED = 42
ALPHA = 0.05
PRE_BUFFER = 100
POST_BUFFER = 100


def _two_sample_ks_per_feature(pre: np.ndarray, post: np.ndarray) -> list[float]:
    return [float(ks_2samp(pre[:, j], post[:, j]).pvalue) for j in range(pre.shape[1])]


def verify(dataset: dict) -> dict:
    name = dataset["name"]
    cfg = dataset["config"]
    print(f"\n--- {name} ---")
    X, _, drift_positions, info = generate_drift_stream(
        cfg, total_size=TOTAL_SIZE, seed=SEED
    )
    transition_width = cfg["params"].get("transition_width", 0)
    half_width = transition_width // 2

    if not drift_positions:
        print("  no drift positions in dataset; skipping")
        return {"name": name, "px_drift": False, "reason": "no drift events"}

    d0 = drift_positions[0]
    pre_end = max(0, d0 - half_width - PRE_BUFFER)
    pre_start = max(0, pre_end - 1500)
    post_start = min(len(X), d0 + half_width + POST_BUFFER)
    post_end = min(len(X), post_start + 1500)

    pre = X[pre_start:pre_end]
    post = X[post_start:post_end]
    print(f"  pre  segment: samples [{pre_start}:{pre_end}]  shape={pre.shape}")
    print(f"  post segment: samples [{post_start}:{post_end}]  shape={post.shape}")

    pvalues = _two_sample_ks_per_feature(pre, post)
    bonferroni = ALPHA / len(pvalues)
    significant = [(j, p) for j, p in enumerate(pvalues) if p < bonferroni]

    print(f"  per-feature KS p-values  (Bonferroni threshold = {bonferroni:.2e}):")
    for j, p in enumerate(pvalues):
        flag = "*" if p < bonferroni else " "
        print(f"    feature {j:2d}: p = {p:.3e} {flag}")

    px_drift = bool(significant)
    print(
        f"  -> P(X) drift detected: {px_drift} "
        f"({len(significant)} of {len(pvalues)} features)"
    )

    return {
        "name": name,
        "px_drift": px_drift,
        "n_features": len(pvalues),
        "n_significant": len(significant),
        "pvalues": pvalues,
    }


def main() -> int:
    print("Phase 0-mini gradual-drift P(X) verification")
    print("=" * 60)
    results = [verify(d) for d in CANDIDATE_DATASETS]

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        verdict = "P(X) DRIFT PRESENT" if r.get("px_drift") else "NO P(X) DRIFT"
        print(
            f"  {r['name']:32s}  {verdict}  "
            f"({r.get('n_significant', 0)}/{r.get('n_features', 0)} features)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
