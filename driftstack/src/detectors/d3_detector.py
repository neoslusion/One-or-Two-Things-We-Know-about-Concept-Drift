
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as roc

class D3Detector:
    """Discriminative Drift Detector (D3) on a sliding window.
    Splits the last 2W samples into halves, trains clf to discriminate, uses AUC.
    Drift score = 1 - AUC. Raise if >= threshold.
    """
    def __init__(self, window=200, clf='logreg', auc_threshold=0.65, name='d3'):
        self.window = window
        self.auc_threshold = auc_threshold
        self.name = name
        if clf == 'logreg':
            self.clf = LogisticRegression(solver="liblinear")
        else:
            self.clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        self.buffer_X = None

    def update(self, x_t):
        # x_t: feature vector
        if self.buffer_X is None:
            self.buffer_X = []
        self.buffer_X.append(x_t)
        if len(self.buffer_X) < 2 * self.window:
            return 0.0
        X = np.array(self.buffer_X[-2*self.window:])
        y = np.ones(2*self.window, dtype=int)
        y[:self.window] = 0

        preds = np.zeros_like(y, dtype=float)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y):
            self.clf.fit(X[tr], y[tr])
            p = self.clf.predict_proba(X[te])[:, 1] if hasattr(self.clf, "predict_proba") else self.clf.decision_function(X[te])
            # map to [0,1] if needed
            if p.ndim == 1:
                p = (p - p.min()) / (p.max() - p.min() + 1e-9)
            preds[te] = p
        auc = roc(y, preds)
        score = 1.0 - auc
        return 1.0 if (1.0 - auc) >= (1.0 - self.auc_threshold) else 0.0
