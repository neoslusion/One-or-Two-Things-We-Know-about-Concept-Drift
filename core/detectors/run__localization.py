import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as roc

from scipy.stats import ortho_group
from scipy.stats import binom
from scipy.stats import norm

import random
from multiprocessing.pool import Pool
from tqdm import tqdm
from copy import deepcopy
import logging

## Drift localization methods. Code based on https://github.com/FabianHinder/Localization-of-Concept-Drift
"""
Computes drift localization based on kdq-Trees (Dasu et al., 2006)
"""
def kdqtree(X,y,y_true, min_size=0.001,min_samps=10):
    if X.shape[1] > 50:
        X = X[:,np.random.choice(range(X.shape[1]),50,replace=False)]
    y_tp = kdqtree0(X,y,y_true, y.sum(),(1-y).sum(), min_size, min_samps)
    y_true, y_pred = np.array([x[0] for x in y_tp]), np.array([x[1] for x in y_tp])
    return roc(~y_true, y_pred)
def kdqtree0(X,y,y_true, y0,y1, min_size=0.1,min_samps=10, dim=0,stop_dim=-1):
    y,y_true = y.astype(bool),y_true.astype(bool)
    if (~y).sum() < min_samps or dim == stop_dim:
        p = y.sum() / y0
        q = (1-y).sum() / y1
        d_kl_v = (p-q)*np.log( (p*(1-q))/(q*(1-p)) ) if (1-p)*(1-q)*p*q != 0 else 1e32 
        return list(zip(list(y_true),len(y_true)*[d_kl_v]))
    else:
        min_, max_ = X[:,dim][y==0].min(),X[:,dim][y==0].max()
        if abs(min_-max_) < min_size:
            return kdqtree0(X,y,y_true, y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1], stop_dim=stop_dim if stop_dim != -1 else dim)
        else:
            s = (min_+max_)/2
            I = X[:,dim] < s; J = np.logical_not(I)
            return   kdqtree0(X[I],y[I],y_true[I], y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1]) \
                   + kdqtree0(X[J],y[J],y_true[J], y0,y1, min_size=min_size,min_samps=min_samps, dim=(dim+1)%X.shape[1])

"""
Computes drift localization based on LDD-DSI (Liu et al., 2017)
"""
def LDD(X,y,y_true, k=0.1, alpha=[0.05]):
    if k <= 1:
        k = int(X.shape[0]*k)
    delta_ = delta(X,y,k)
    
    return roc(~y_true, np.abs(delta_))
    
def delta(X,y,k):
    y = y.astype(int)
    x = KNeighborsClassifier(n_neighbors=k).fit(X,y).predict_proba(X)[np.vstack( (y==0,y==1) ).T]
    x *= 1-2*1e-16;x += 1e-16
    return x/(1-x)-1

"""
Computes drift localization based on knn. Optimal threashold is defined in a post hoc fashion (BASELINE ONLY!.
"""
def knn_post_hoc(X,y,y_true, k=8):
    if abs(y.mean()-0.5) > 0.05:
        logging.warning("knn assumes P[t = 0] ~ 0.50, current is %.3f"%y.mean())
    sigma = 2*np.abs(KNeighborsClassifier(n_neighbors=k).fit(X,y).predict_proba(X)[:,0]-0.5)
    return roc(~y_true, sigma)

"""
Simple drift localization. H0 for kNN and approximative/virtual leaf for RF.
"""
def simple_drift_localization(X,y,y_true,model,k):
    res = model.fit(X,y).predict_proba(X)[:,1]
    y_mean = y.mean()
    p0 = binom.cdf( res*k, k, y_mean )
    return roc(y_true, np.vstack( (p0,1-p0) ).min(axis=0))
#######



def run_exp(inten, lam, samp, dim, exp_id):
    X = np.random.random(size=(samp,dim)) - 0.5
    t = int(X.shape[0]/2)
    X[:t,:2] -= inten/2
    X[t:,:2] += inten/2
    d = np.abs(X[:,:2]).max(axis=1)
    
    X = X @ (lam * ortho_group.rvs(X.shape[1]) + (1-lam)*np.eye(X.shape[1]))
    
    y = np.zeros(X.shape[0]); y[t:] = 1
    y_true = d < (1-inten)/2
    
    if 2*abs(y_true.mean()-0.5) > 0.95:
        return []
    res = [{"exp_id": exp_id,
            "inten":inten, 
            "lam": lam, 
            "samp": samp, 
            "dim": dim, 
            "method": method, 
            "value": val} for method,val in zip(["kdqTree","LDD","DL"],
                                                [kdqtree(X,y,y_true), 
                                                 LDD(X,y,y_true), 
                                                 simple_drift_localization(X,y,y_true,RandomForestClassifier(criterion='gini',min_samples_leaf=10),k=10)])]
    return res


intens = [0.01,0.025,0.05,0.1]
lams   = list(np.linspace(0,1,10))
dims   = [2,5,10,25,50]
samps  = [25,50,100,150,250,500,750]

tasks = []
for inten in intens:
    for lam in lams:
        for dim in dims:
            for samp in samps:
                tasks.append({"inten": inten, "lam": lam, "samp": samp, "dim": dim})

try:
    df = pd.read_pickle("localization.pkl.gzip")
    add_to_id = df["exp_id"].max()+1
except:
    df = None
    add_to_id = 0
    
tasks = [deepcopy(t) for t in 500*tasks]
for i,t in enumerate(tasks):
    t["exp_id"] = i + add_to_id
random.shuffle(tasks)


def run_exp0(x):
    return run_exp(**x)

with Pool(1) as pool:
    res = []
    for r in tqdm(pool.imap(run_exp0,tasks), total=len(tasks)):
         res += r
         if len(res) > 50000:
              df = pd.concat( (df, pd.DataFrame(res) ) )
              df.to_pickle("out__localization.pkl.xz")
              res = []
df = pd.concat( (df, pd.DataFrame(res) ) )
df.to_pickle("out__localization.pkl.xz")
