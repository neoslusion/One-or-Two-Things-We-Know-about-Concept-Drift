import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score as roc
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from copy import deepcopy

import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm
import os
import random
import time


from gen_data import gen_random

from mmd import mmd
from ks import ks
from dawidd import dawidd
from shape_dd import shape
from kernel_dd import kernel_dd
from d3 import d3


def batch(X, chunk_size, overlap=100):
    shift = chunk_size-overlap
    res = []
    for i in range(int(X.shape[0]/(shift))-int(chunk_size/shift)+1):
        res.append( X[i*shift : min(chunk_size + i*shift, X.shape[0])] )
    return res

def time_it(f,x):
    t0 = time.time()
    y = f(x)
    t1 = time.time()
    return t1-t0, y

def run_exp(task, value, chunk_size, dist, alt, exp_id):
    X,y=gen_random(**dict([("dist",dist),("alt",alt),(task,value)  ,  ("length",750),("min_dist",10)]))
    
    t0 = time.time()
    shp = shape(X, 50,chunk_size, 2500)[:,2]
    t1 = time.time()
    kdd = kernel_dd(X)
    t2 = time.time()
    
    res = []
    batches = batch(np.arange(X.shape[0]), chunk_size=chunk_size)
    batch_count = len(batches)
    for b in batches:
        x_ = X[b]
        drift_score = (y[b][None,:] != y[b][:,None]).sum()/( b.shape[0]*(b.shape[0]-1) )
        res += [{"task":task, "value":value, "dist":dist, "alt":alt, "drift_score":drift_score, "exp_id":exp_id, "method":method, "estimate":estimate, "chunk_size": chunk_size, "comp_time": comp_time} for method,(comp_time,estimate) in 
               {
                   "mmd":          time_it(lambda x:float("%.5f"%mmd(x)[1]),x_),
                   "d3 (lin)":     time_it(lambda x:float("%.5f"%d3(x)),x_),
                   "d3 (ET)":      time_it(lambda x:float("%.5f"%d3(x, ExtraTreesClassifier(max_depth=5))),x_),
                   "d3 (RF)":      time_it(lambda x:float("%.5f"%d3(x, RandomForestClassifier(max_depth=5))),x_),
                   "d3 (kNN)":     time_it(lambda x:float("%.5f"%d3(x, KNeighborsClassifier())),x_),
                   "ks":           time_it(lambda x:float("%.5f"%ks(x)),x_),
                   "dawidd (rbf)": time_it(lambda x:float("%.5f"%dawidd(x,"rbf")[1]),x_),
                   "kdd (loc)":    time_it(lambda x:float("%.5f"%kernel_dd(x).min()),x_),
                   "kdd (glob)":   ((t2-t1)/batch_count,float("%.5f"%kdd[b].min())),
                   "shape":        ((t1-t0)/batch_count,float("%.5f"%shp[b].min()))
               }.items()]
    return res

tasks = []
for task,values in {"dims":[2,3,4,5,10,15,20,30,50,100,150,200], 
                    "number":[1,2,3,4,5,10], 
                    "intens":np.linspace(0,0.25,8+1)[1:]
                   }.items():
    for value in values:
        for dist in ["unif","gauss","dubi"]:
            for alt in [True,False] if dist == "unif" else [True]:
                for chunk_size in [150,250]:
                    tasks.append({"task":task,"value":value,"alt":alt,"dist":dist, "chunk_size":chunk_size})

random.shuffle(tasks)
tasks = [deepcopy(t) for t in 100*tasks]
for i,t in enumerate(tasks):
    t["exp_id"] = i

def run_exp0(x):
    return run_exp(**x)

cpu_count = 1

print("Using %i CPUs"%cpu_count)


pool = Pool(cpu_count)
res, df = [], None
t0 = time.time()
for r in tqdm(pool.imap(run_exp0,tasks), total=len(tasks)):
    res += r
    if time.time()-t0 > 5*60:
        df = pd.concat( (df, pd.DataFrame(res) ) ) if df is not None else pd.DataFrame(res)
        df.to_pickle("out__detection.pkl.xz")
        res = []
        t0 = time.time()
df = pd.concat( (df, pd.DataFrame(res) ) )
df.to_pickle("out__detection.pkl.xz")

