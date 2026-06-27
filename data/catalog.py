"""
Dataset catalog for drift detection benchmarks.

Contains the DATASET_CATALOG dictionary defining all available benchmark datasets
and helper functions to filter enabled datasets.

Notes on `ground_truth_type`:
    - "exact":     synthetic stream with known drift positions
    - "estimated": semi-real / heuristic positions
    - "none":      no ground-truth drift positions
"""

DATASET_CATALOG = {
    # ========================================================================
    # SUDDEN DRIFT (P(X) change) — main benchmark contributors
    # ========================================================================
    "stagger": {
        "enabled": True,
        "type": "stagger",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {},
    },
    "stagger_recurrent_explicit": {
        # Dedicated 4-concept STAGGER variant providing clean recurrent-drift
        # coverage (events 0-2 sudden, 3-9 recurrent).  See
        # `generate_stagger_recurrent_explicit_stream` for the concept rules.
        "enabled": True,
        "type": "stagger_recurrent_explicit",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {},
    },
    "gen_random_mild": {
        "enabled": True,
        "type": "gen_random",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"dims": 5, "intens": 0.125, "dist": "unif", "alt": False},
    },
    "gen_random_moderate": {
        "enabled": True,
        "type": "gen_random",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"dims": 5, "intens": 0.25, "dist": "unif", "alt": False},
    },
    "gen_random_severe": {
        "enabled": True,
        "type": "gen_random",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"dims": 5, "intens": 1, "dist": "unif", "alt": True},
    },
    "gen_random_ultra_severe": {
        "enabled": True,
        "type": "gen_random",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"dims": 5, "intens": 2, "dist": "unif", "alt": True},
    },
    "gaussian_shift_moderate": {
        "enabled": True,
        "type": "gaussian_shift",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {
            "n_features": 10,
            "shift_magnitude": 1.5,
            "noise_percentage": 0.05,
        },
    },

    # ========================================================================
    # SUPPLEMENTARY: P(Y|X)-only concept drift (do NOT count toward main benchmark)
    # Kept to demonstrate that unsupervised P(X) detectors cannot detect Y|X drift.
    # ========================================================================
    "standard_sea": {
        "enabled": True,
        "type": "standard_sea",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "drift_category": "concept_drift_only",
        "params": {},
    },
    "hyperplane": {
        "enabled": True,
        "type": "hyperplane",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "drift_category": "concept_drift_only",
        "params": {"n_features": 10},
    },

    # ========================================================================
    # GRADUAL DRIFT
    # circles_gradual was the historical gradual benchmark.  It is kept
    # DISABLED here because Phase 0-mini verification (see
    # scripts/verify_gradual_px_drift.py) shows that the dataset only
    # changes P(Y|X), not P(X), and is therefore invisible to unsupervised
    # P(X) detectors.  gaussian_gradual_synthetic is enabled in its place
    # to provide genuine gradual P(X) coverage with a controllable
    # transition width.
    # ========================================================================
    "circles_gradual": {
        "enabled": False,
        "type": "circles_gradual",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"transition_width": 400},
    },
    "gaussian_gradual_synthetic": {
        "enabled": True,
        "type": "gaussian_gradual",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {
            "n_features": 10,
            "shift_magnitude": 1.5,
            "transition_width": 400,
            "noise_percentage": 0.05,
        },
    },

    # ========================================================================
    # SEMI-REAL: real features with synthetic drift at known positions
    # ========================================================================
    "electricity_semisynthetic": {
        "enabled": True,
        "type": "electricity_semisynthetic",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {},
    },

    # ========================================================================
    # STATIONARY: false-positive / Type-I error calibration
    # ========================================================================
    "stagger_none": {
        "enabled": True,
        "type": "stagger",
        "n_drift_events": 0,
        "ground_truth_type": "exact",
        "params": {},
    },

    # ========================================================================
    # COMPLEX DISTRIBUTIONS: blip / discrete features
    # ========================================================================
    "rbfblips": {
        "enabled": True,
        "type": "rbfblips",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"n_centroids": 50, "n_features": 10},
    },
    "led_abrupt": {
        "enabled": True,
        "type": "led_abrupt",
        "n_drift_events": 10,
        "ground_truth_type": "exact",
        "params": {"has_noise": False},
    },
}


def get_enabled_datasets():
    """
    Get list of enabled datasets from the catalog.

    Returns:
        list: List of (name, config) tuples for enabled datasets
    """
    return [
        (name, config)
        for name, config in DATASET_CATALOG.items()
        if config["enabled"]
    ]


def get_datasets_by_type():
    """
    Categorize enabled datasets by drift type.

    Returns:
        dict: Keys 'sudden', 'gradual', 'realworld', 'stationary',
              'concept_drift_only'.
    """
    enabled = [k for k, v in DATASET_CATALOG.items() if v["enabled"]]

    def _drift_class(k: str) -> str:
        cfg = DATASET_CATALOG[k]
        if cfg.get("drift_category") == "concept_drift_only":
            return "concept_drift_only"
        if cfg["n_drift_events"] == 0 or "none" in k:
            return "stationary"
        if "gradual" in k:
            return "gradual"
        if "electricity" in k or "covertype" in k:
            return "realworld"
        return "sudden"

    classes = {
        "sudden": [],
        "gradual": [],
        "realworld": [],
        "stationary": [],
        "concept_drift_only": [],
    }
    for k in enabled:
        classes[_drift_class(k)].append(k)
    return classes
