"""
Adaptation strategies for different drift types.

Each strategy implements a specific model update approach based on drift characteristics.
Uses sklearn batch learning to match MultiDetectorEvaluation.ipynb notebook.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List, Callable
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def log(msg: str):
    print(f"[Strategy] {msg}", flush=True)


def err(msg: str):
    print(f"[Strategy][ERROR] {msg}", flush=True)


def adapt_sudden_drift(model_factory: Callable, X: np.ndarray, y: Optional[np.ndarray],
                       feature_names: Optional[List[str]] = None, allow_unlabeled: bool = False):
    """
    SUDDEN drift: Full model reset and retrain (sklearn batch learning).
    
    Matches MultiDetectorEvaluation.ipynb approach:
    - Create fresh sklearn Pipeline
    - Train on post-drift data using .fit()
    - Abrupt change - best to start fresh
    """
    log("SUDDEN drift → Full model reset and retrain")
    new_model = model_factory()

    if y is not None:
        # Check for single-class issue
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            log(f"  WARNING: Only {len(unique_classes)} class in data, skipping retrain")
            return model_factory()  # Return fresh untrained model
        
        # Batch training with sklearn
        new_model.fit(X, y)
        log(f"  Retrained on {len(X)} samples using sklearn .fit()")
    elif allow_unlabeled:
        log(f"  WARNING: Cannot train classifier without labels - returning untrained model")

    return new_model


def adapt_incremental_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                            feature_names: Optional[List[str]] = None):
    """
    INCREMENTAL drift: Batch retrain on recent data (sklearn approach).
    
    Note: sklearn doesn't support true incremental learning for LogisticRegression pipeline.
    Strategy: Retrain on the drift window data (similar to sudden but keeps context).
    Monotonic progression - retrain on recent samples.
    """
    log("INCREMENTAL drift → Retrain on drift window")
    
    if y is not None:
        # Check for single-class issue
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            log(f"  WARNING: Only {len(unique_classes)} class in data, skipping retrain")
            return model
        
        # Batch retrain on drift window
        model.fit(X, y)
        log(f"  Retrained model on {len(X)} recent samples")
    else:
        log("  WARNING: Cannot retrain without labels")
    
    return model


def adapt_gradual_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                        feature_names: Optional[List[str]] = None):
    """
    GRADUAL drift: Retrain on recent samples (sklearn batch approach).
    
    Non-monotonic with oscillations - focus on recent 50% of samples.
    Note: sklearn doesn't support sample weighting in Pipeline easily,
    so we use a subset strategy instead.
    """
    log("GRADUAL drift → Retrain on recent samples")

    if y is None:
        log("  WARNING: Cannot retrain without labels")
        return model

    # Focus on recent half (gradual drift still evolving)
    n = len(X)
    cutoff = n // 2
    X_recent = X[cutoff:]
    y_recent = y[cutoff:]

    # Check for single-class issue (common with drift data)
    unique_classes = np.unique(y_recent)
    if len(unique_classes) < 2:
        log(f"  WARNING: Only {len(unique_classes)} class in recent samples, using full window")
        X_recent = X
        y_recent = y
        unique_classes = np.unique(y_recent)
        if len(unique_classes) < 2:
            log(f"  WARNING: Still single class, skipping retrain")
            return model

    # Batch retrain on recent samples
    model.fit(X_recent, y_recent)
    log(f"  Retrained on recent {len(X_recent)} samples (50% of window)")
    return model


def compute_distribution_similarity(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute distribution similarity using average KS distance across features.
    Returns similarity score [0, 1] where 0 = identical, 1 = completely different.
    """
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n_features = min(X1.shape[1], X2.shape[1])
    ks_distances = []

    for j in range(n_features):
        stat, _ = ks_2samp(X1[:, j], X2[:, j])
        ks_distances.append(stat)

    return float(np.mean(ks_distances))


def find_best_cached_model(X: np.ndarray, cache_dir: Path, similarity_threshold: float = 0.15):
    """
    Find best cached model by comparing distribution similarity.

    Returns:
        tuple: (cache_path, similarity_score) or (None, None) if no good match
    """
    cached_models = list(cache_dir.glob("model_*.npz"))

    if not cached_models:
        return None, None

    best_match = None
    best_similarity = float('inf')

    for cache_path in cached_models:
        try:
            # Load cached distribution snapshot
            data = np.load(cache_path, allow_pickle=True)
            cached_X = data.get("X_snapshot")

            if cached_X is None:
                continue

            # Compute similarity
            similarity = compute_distribution_similarity(X, cached_X)

            if similarity < best_similarity:
                best_similarity = similarity
                best_match = cache_path

        except Exception as e:
            err(f"Error loading cache {cache_path.name}: {e}")
            continue

    if best_match and best_similarity < similarity_threshold:
        return best_match, best_similarity

    return None, None


def adapt_recurrent_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                          feature_names: Optional[List[str]] = None,
                          cache_dir: Optional[Path] = None,
                          similarity_threshold: float = 0.15):
    """
    RECURRENT drift: Use cached sklearn model if pattern repeats.
    
    Return to previous distribution - try to reuse old models.
    Fine-tune cached model with sklearn batch learning.
    """
    log("RECURRENT drift → Checking for cached model")

    if cache_dir is None or not cache_dir.exists():
        log("  No cache directory → fallback to batch retrain")
        return adapt_incremental_drift(model, X, y, feature_names)

    # Find best matching cached model
    cache_path, similarity = find_best_cached_model(X, cache_dir, similarity_threshold)

    if cache_path:
        log(f"  Found cached model: {cache_path.stem} (similarity: {similarity:.4f})")
        try:
            # Load model pickle (not npz)
            model_pkl = cache_path.with_suffix('.pkl')
            if model_pkl.exists():
                import pickle
                with open(model_pkl, "rb") as f:
                    cached_model = pickle.load(f)

                # Fine-tune cached model with sklearn
                if y is not None:
                    cached_model.fit(X, y)
                    log(f"  Loaded cached model and fine-tuned with {len(X)} samples")
                else:
                    log(f"  Loaded cached model without fine-tuning (no labels)")

                return cached_model
        except Exception as e:
            err(f"  Failed to load cached model: {e}")

    log("  No suitable cached model → fallback to batch retrain")
    return adapt_incremental_drift(model, X, y, feature_names)


def adapt_blip_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                     feature_names: Optional[List[str]] = None):
    """
    BLIP: Minimal update (sklearn batch approach).
    
    Very short temporary anomaly - likely noise, don't overreact.
    Conservative strategy: retrain on small subset or skip entirely.
    """
    log("BLIP drift → Minimal update (temporary noise)")

    if y is not None and len(X) > 0:
        # Very conservative: only if we have reasonable sample size
        if len(X) >= 10:
            n_samples = min(10, len(X))
            model.fit(X[:n_samples], y[:n_samples])
            log(f"  Conservative retrain with {n_samples} samples only")
        else:
            log(f"  Ignoring blip - too few samples ({len(X)})")
    else:
        log("  Ignoring unlabeled blip")

    return model


def cache_model_with_distribution(model, X: np.ndarray, drift_idx: int, cache_dir: Path):
    """
    Cache model with distribution snapshot for pattern matching.
    Saves both model (.pkl) and distribution (.npz).
    """
    import pickle

    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(__import__('time').time())
    base_name = f"model_{drift_idx}_{timestamp}"

    # Save model
    model_path = cache_dir / f"{base_name}.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save distribution snapshot for similarity comparison
        snapshot_path = cache_dir / f"{base_name}.npz"
        np.savez(snapshot_path, X_snapshot=X)

        log(f"  Cached model and distribution: {base_name}")
    except Exception as e:
        err(f"  Failed to cache model: {e}")
