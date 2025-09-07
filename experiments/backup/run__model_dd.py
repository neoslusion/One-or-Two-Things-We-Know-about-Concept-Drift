import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.base import clone as sk_cp
from sklearn.metrics import roc_auc_score as roc
from copy import deepcopy
from multiprocessing.pool import Pool
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GridSearchCV
import random
import time

## Use gen data with different default parameters
import gen_data 
def gen_random(number=1, dims=5, intens=0.5, dist="unif", alt=False, length=750, min_dist=10, min_dist_border=100):
    return gen_data.gen_random(number, dims, intens, dist, alt, length, min_dist, min_dist_border)
    

def eval_model(X,y,model,parameters = None, split=100):
    y_true = y[split:] == 0
    
    model = sk_cp(model)
    if parameters is not None:
        model = GridSearchCV(model, parameters).fit(X[:split]).best_estimator_
    model.fit(X[:split])
    
    if hasattr(model,"score_samples"):
        y_pred = model.score_samples(X[split:])
    elif hasattr(model,"predict"):
        y_pred = model.predict(X[split:])
    else:
        raise ValueError()
    
    return roc(y_true,y_pred), y_pred

def run_exp(exp_id, dist, task, value):
    X,y=gen_random(**dict([("dist",dist),("alt",True), (task, value), ("length",750),("min_dist",100)]))
    
    return [{"task":task, "value":value, "dist":dist, "exp_id":exp_id, "method":method, "estimate":eval_model(X,y,model,parameters)[0]} for method,(model,parameters) in
                 {
                     "BayesianGaussianMixture": (BayesianGaussianMixture(n_components=10, max_iter=1000),None),
                     "GaussianMixture": (GaussianMixture(), {"n_components": range(1,10)}),
                     "KernelDensity": (KernelDensity(),{"kernel": ["gaussian", "exponential"], "bandwidth": np.logspace(-1,np.log(5)/np.log(10),20)}), 
                     "OneClassSVM": (OneClassSVM(),None),
                     "OneClassSVM (line)": (OneClassSVM(kernel="linear"),None),
                     "OneClassSVM (poly)": (OneClassSVM(kernel="poly"),None),
                     "OneClassSVM (sigm)": (OneClassSVM(kernel="sigmoid"),None),
                     "IsolationForest": (IsolationForest(),None),
                     "LocalOutlierFactor (p: 1)": (LocalOutlierFactor(novelty=True, p=1),None),
                     "LocalOutlierFactor": (LocalOutlierFactor(novelty=True),None),
                     "LocalOutlierFactor (k: 30)": (LocalOutlierFactor(novelty=True, n_neighbors=30),None),
                     "LocalOutlierFactor (k: 10)": (LocalOutlierFactor(novelty=True, n_neighbors=10),None),
                     "LocalOutlierFactor (k: 5)": (LocalOutlierFactor(novelty=True, n_neighbors=5),None),
                 }.items()]

tasks = []

for task,values in {"dims":[2,3,4,5,10,15,20,30,50,100,150,200], 
                    "number":[1,5,10], 
                    "intens":list(np.linspace(0,0.25,8+1)[1:])+list(np.linspace(0.25,1,8+1)[1:])
                   }.items():
    for value in values:
        for dist in ["unif","gauss","dubi"]:
            tasks.append({"task":task,"value":value,"dist":dist})

random.shuffle(tasks)
tasks = [deepcopy(t) for t in 500*tasks]
for i,t in enumerate(tasks):
    t["exp_id"] = i 



def run_exp0(x):
    return run_exp(**x)

with Pool(1) as pool:
    res = []
    t0 = time.time()
    for r in tqdm(pool.imap(run_exp0,tasks), total=len(tasks)):
        res += r
        if time.time()-t0 > 5*60:
            pd.DataFrame(res).to_pickle("out__model_dd.pkl.xz")
            t0 = time.time()
    pd.DataFrame(res).to_pickle("out__model_dd.pkl.xz")
