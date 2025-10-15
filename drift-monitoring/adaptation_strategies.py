"""
Adaptation strategies for different drift types.

Each strategy implements a specific model update approach based on drift characteristics.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List
from scipy.stats import ks_2samp


def log(msg: str):
    print(f"[Strategy] {msg}", flush=True)


def err(msg: str):
    print(f"[Strategy][ERROR] {msg}", flush=True)


def _to_records(X: np.ndarray, feature_names: Optional[List[str]] = None):
    """Convert numpy array to River record format (list of dicts)."""
    n, d = X.shape
    if feature_names is None:
        feature_names = [str(j) for j in range(d)]
    return [{feature_names[j]: float(X[i, j]) for j in range(d)} for i in range(n)]


def learn_many(model, X: np.ndarray, y: Optional[np.ndarray], feature_names: Optional[List[str]] = None):
    """Online update for River model."""
    if y is None:
        log("No labels - updating scaler statistics only")
        Xrecs = _to_records(X, feature_names)
        for xi in Xrecs:
            _ = model.transform_one(xi)
        return model

    Xrecs = _to_records(X, feature_names)
    for xi, yi in zip(Xrecs, y):
        model = model.learn_one(xi, yi)
    return model


def adapt_sudden_drift(model_factory, X: np.ndarray, y: Optional[np.ndarray],
                       feature_names: Optional[List[str]] = None, allow_unlabeled: bool = False):
    """
    SUDDEN drift: Full model reset and retrain.
    Abrupt change - best to start fresh.
    """
    log("SUDDEN drift → Full model reset and retrain")
    new_model = model_factory()

    if y is not None:
        new_model = learn_many(new_model, X, y, feature_names)
        log(f"  Retrained on {len(X)} samples")
    elif allow_unlabeled:
        new_model = learn_many(new_model, X, y, feature_names)
        log(f"  Updated scaler with {len(X)} unlabeled samples")

    return new_model


def adapt_incremental_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                            feature_names: Optional[List[str]] = None):
    """
    INCREMENTAL drift: Gradual online updates.
    Monotonic progression - keep existing model and adapt gradually.
    """
    log("INCREMENTAL drift → Gradual online updates")
    model = learn_many(model, X, y, feature_names)
    log(f"  Applied incremental updates with {len(X)} samples")
    return model


def adapt_gradual_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                        feature_names: Optional[List[str]] = None):
    """
    GRADUAL drift: Weighted updates with recent sample priority.
    Non-monotonic with oscillations - use weighted learning.
    """
    log("GRADUAL drift → Weighted updates (recent samples prioritized)")

    if y is None:
        model = learn_many(model, X, y, feature_names)
        log(f"  Updated scaler with {len(X)} samples")
        return model

    n = len(X)
    Xrecs = _to_records(X, feature_names)

    # Weighted learning: recent samples get higher weight
    for i, (xi, yi) in enumerate(zip(Xrecs, y)):
        weight = (i + 1) / n
        if weight > 0.5:  # Focus on recent half
            model = model.learn_one(xi, yi)

    log(f"  Applied weighted updates focusing on recent {n//2} samples")
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
    RECURRENT drift: Use cached model if pattern repeats.
    Return to previous distribution - try to reuse old models.
    """
    log("RECURRENT drift → Checking for cached model")

    if cache_dir is None or not cache_dir.exists():
        log("  No cache directory → fallback to incremental")
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

                # Fine-tune cached model
                if y is not None:
                    cached_model = learn_many(cached_model, X, y, feature_names)
                    log(f"  Loaded cached model and fine-tuned with {len(X)} samples")

                return cached_model
        except Exception as e:
            err(f"  Failed to load cached model: {e}")

    log("  No suitable cached model → fallback to incremental")
    return adapt_incremental_drift(model, X, y, feature_names)


def adapt_blip_drift(model, X: np.ndarray, y: Optional[np.ndarray],
                     feature_names: Optional[List[str]] = None):
    """
    BLIP: Minimal update.
    Very short temporary anomaly - likely noise, don't overreact.
    """
    log("BLIP drift → Minimal update (temporary noise)")

    if y is not None:
        # Conservative update with small subset
        n_samples = min(5, len(X))
        model = learn_many(model, X[:n_samples], y[:n_samples], feature_names)
        log(f"  Conservative update with {n_samples} samples only")
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
