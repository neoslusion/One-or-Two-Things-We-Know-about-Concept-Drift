"""
Dataset catalog for drift detection benchmarks.

Contains the DATASET_CATALOG dictionary defining all available benchmark datasets
and helper functions to filter enabled datasets.
"""

DATASET_CATALOG = {
    # ========================================================================
    # SUDDEN DRIFT DATASETS (8 datasets)
    # Classic benchmarks with abrupt concept switches
    # ground_truth_type: "exact" = known positions, "estimated" = heuristic, "none" = unknown
    # ========================================================================
    "standard_sea": {
        "enabled": True,
        "type": "standard_sea",
        "n_drift_events": 10,
        "ground_truth_type": "exact",  # Synthetic - exact drift positions known
        "params": {}
    },
    "enhanced_sea": {
        "enabled": False,
        "type": "enhanced_sea",
        "n_drift_events": 10,
        "params": {
            "scale_factors": (1.8, 1.5, 2.0),
            "shift_amounts": (5.0, 4.0, 8.0)
        }
    },
    "stagger": {
        "enabled": True,
        "type": "stagger",
        "n_drift_events": 10,
        "ground_truth_type": "exact",  # Synthetic - exact drift positions known
        "params": {}
    },
    "hyperplane": {
        "enabled": True,
        "type": "hyperplane",
        "n_drift_events": 10,
        "ground_truth_type": "exact",  # Synthetic - exact drift positions known
        "params": {
            "n_features": 3
        }
    },
    "gen_random_mild": {
        "enabled": False,
        "type": "gen_random",
        "n_drift_events": 10,
        "params": {
            "dims": 5,
            "intens": 0.125,
            "dist": "unif",
            "alt": False
        }
    },
    "gen_random_moderate": {
        "enabled": False,
        "type": "gen_random",
        "n_drift_events": 10,
        "params": {
            "dims": 5,
            "intens": 0.25,
            "dist": "unif",
            "alt": False
        }
    },
    "gen_random_severe": {
        "enabled": False,
        "type": "gen_random",
        "n_drift_events": 10,
        "params": {
            "dims": 5,
            "intens": 1,
            "dist": "unif",
            "alt": True
        }
    },
    "gen_random_ultra_severe": {
        "enabled": False,
        "type": "gen_random",
        "n_drift_events": 10,
        "params": {
            "dims": 5,
            "intens": 2,
            "dist": "unif",
            "alt": True
        }
    },

    # ========================================================================
    # GRADUAL DRIFT DATASETS (4 datasets)
    # Smooth blending transitions between concepts
    # ========================================================================

    "sea_gradual": {
        "enabled": True,  # ENABLED: Representative gradual drift dataset
        "type": "sea_gradual",
        "n_drift_events": 10,
        "ground_truth_type": "exact",  # Synthetic - exact transition start positions known
        "params": {
            "transition_width": 450  # OPTIMIZED: 50% of segment (909 samples)
        }
    },
    "hyperplane_gradual": {
        "enabled": False,
        "type": "hyperplane_gradual",
        "n_drift_events": 10,
        "params": {
            "n_features": 10  # Continuous drift (no discrete transition)
        }
    },
    "agrawal_gradual": {
        "enabled": False,
        "type": "agrawal_gradual",
        "n_drift_events": 10,
        "params": {
            "transition_width": 450  # OPTIMIZED: 50% of segment (909 samples)
        }
    },
    "circles_gradual": {
        "enabled": False,
        "type": "circles_gradual",
        "n_drift_events": 10,
        "params": {
            "transition_width": 400  # OPTIMIZED: 44% of segment (more stable)
        }
    },

    # ========================================================================
    # INCREMENTAL DRIFT DATASETS (2 datasets) - MOA Standard
    # Continuous cluster boundary movement
    # ========================================================================

    "rbf_slow": {
        "enabled": True,  # ENABLED: Representative incremental drift dataset
        "type": "rbf",
        "n_drift_events": 10,
        "ground_truth_type": "estimated",  # Continuous drift - positions are estimates
        "params": {
            "n_centroids": 50,    # MOA standard
            "speed": 0.0001       # Slow continuous drift
        }
    },
    "rbf_fast": {
        "enabled": False,
        "type": "rbf",
        "n_drift_events": 10,
        "params": {
            "n_centroids": 50,    # MOA standard
            "speed": 0.001        # Fast continuous drift (10× faster)
        }
    },

    # ========================================================================
    # REAL-WORLD DATASETS (1 dataset)
    # Natural concept drift from real-world processes
    # NO GROUND TRUTH - Use for qualitative analysis only
    # ========================================================================

    "electricity": {
        "enabled": False,
        "type": "electricity",
        "n_drift_events": 5,      # Estimated (no ground truth available)
        "params": {}
    },
    "electricity_sorted": {
        "enabled": True,
        "type": "electricity_sorted",
        "n_drift_events": 5,
        "ground_truth_type": "estimated",  # Semi-real: drift positions are heuristic
        "params": {
            "sort_feature": "nswdemand"
        }
    },

    # ========================================================================
    # SEMI-REAL DATASETS - Controlled Drift from Real Data
    # Real-world features with known drift positions (sorted by feature)
    # ========================================================================

    "covertype_sorted": {
        "enabled": False,  # DISABLED: Covtype not available in river library
        "type": "covertype_sorted",
        "n_drift_events": 5,      # Controllable - adjust as needed
        "ground_truth_type": "estimated",  # Semi-real: drift positions are heuristic
        "params": {
            "sort_feature": "Elevation"  # Creates natural drift by terrain
        }
    },

    # ========================================================================
    # STATIONARY DATASETS (2 datasets) - False Positive Analysis
    # No drift - for statistical calibration validation
    # ========================================================================

    "stagger_none": {
        "enabled": True,  # ENABLED: False positive calibration (no drift baseline)
        "type": "stagger",
        "n_drift_events": 0,      # NO DRIFT - stationary
        "ground_truth_type": "exact",  # Synthetic - known to have NO drift
        "params": {}
    },
    "gen_random_none": {
        "enabled": False,
        "type": "gen_random",
        "n_drift_events": 0,      # NO DRIFT - stationary
        "params": {
            "dims": 5,
            "intens": 0,          # Zero intensity = no drift
            "dist": "unif",
            "alt": False
        }
    },

    # ========================================================================
    # SINE FAMILY: Classification Reversal + Noise Robustness Tests
    # ========================================================================
    "sine1": {
        "enabled": False,
        "type": "sine1",
        "n_drift_events": 10,
        "params": {}
    },
    "sine2": {
        "enabled": False,
        "type": "sine2",
        "n_drift_events": 10,
        "params": {}
    },
    "sinirrel1": {
        "enabled": False,
        "type": "sinirrel1",
        "n_drift_events": 10,
        "params": {}
    },
    "sinirrel2": {
        "enabled": False,
        "type": "sinirrel2",
        "n_drift_events": 10,
        "params": {}
    },

    # ========================================================================
    # RBF AND LED: Complex Distributions
    # ========================================================================
    "rbfblips": {
        "enabled": True,
        "type": "rbfblips",
        "n_drift_events": 10,
        "params": {
            "n_centroids": 50,
            "n_features": 10
        }
    },
    "led_abrupt": {
        "enabled": False,
        "type": "led_abrupt",
        "n_drift_events": 10,
        "params": {
            "has_noise": False
        }
    },
}


def get_enabled_datasets():
    """
    Get list of enabled datasets from the catalog.

    Returns:
        list: List of (name, config) tuples for enabled datasets
    """
    return [(name, config) for name, config in DATASET_CATALOG.items()
            if config['enabled']]


def get_datasets_by_type():
    """
    Categorize enabled datasets by drift type.

    Returns:
        dict: Dictionary with keys 'sudden', 'gradual', 'incremental', 'realworld', 'stationary'
    """
    enabled = [k for k, v in DATASET_CATALOG.items() if v['enabled']]

    return {
        'sudden': [k for k in enabled if 'gradual' not in k and 'rbf' not in k
                   and 'electricity' not in k and 'covertype' not in k and 'none' not in k],
        'gradual': [k for k in enabled if 'gradual' in k],
        'incremental': [k for k in enabled if 'rbf' in k],
        'realworld': [k for k in enabled if 'electricity' in k or 'covertype' in k],
        'stationary': [k for k in enabled if 'none' in k],
    }

