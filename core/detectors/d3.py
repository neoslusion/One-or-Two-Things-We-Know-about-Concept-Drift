import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as roc

## D3 Drift Detector. Code Based on https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift/blob/master/D3.py

def d3(X,clf = LogisticRegression(solver='liblinear')):
    y = np.ones(X.shape[0])
    y[:int(X.shape[0]/2)] = 0
    
    predictions = np.zeros(y.shape)
    
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = probs
    auc_score = roc(y, predictions)
    
    return 1 - auc_score
