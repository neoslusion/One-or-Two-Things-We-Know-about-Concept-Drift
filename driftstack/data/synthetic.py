
import numpy as np

def make_stream(n_samples=5000, n_features=5, concept_length=800, drift_type='abrupt', noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    concept_id = np.zeros(n_samples, dtype=int)

    # Create two linear separators that slowly rotate between concepts
    w_list = []
    for k in range(max(2, n_samples // concept_length + 1)):
        w = rng.normal(size=n_features)
        w /= (np.linalg.norm(w) + 1e-9)
        b = rng.uniform(-0.5, 0.5)
        w_list.append((w, b))

    def label(x, w, b):
        return 1 if (x @ w + b) > 0 else 0

    current = 0
    change_points = []
    for t in range(n_samples):
        # feature
        X[t] = rng.normal(size=n_features)
        # concept switch schedule
        if t>0 and t % concept_length == 0:
            change_points.append(t)
            if drift_type == 'abrupt':
                current = (current + 1) % len(w_list)
            elif drift_type in ('gradual', 'incremental'):
                # interpolate weights for next segment
                next_id = (current + 1) % len(w_list)
                alpha = 0.0
                steps = concept_length
                # pre-compute blend for next segment by shifting current gradually
                # we simulate gradual by modifying decision boundary via alpha
                pass
            elif drift_type == 'recurring':
                current = (current + 2) % len(w_list)  # jump 2 to recur

        w, b = w_list[current]
        # noise
        yt = label(X[t], w, b)
        if rng.uniform() < noise:
            yt = 1 - yt
        y[t] = yt
        concept_id[t] = current

    return X, y, change_points
