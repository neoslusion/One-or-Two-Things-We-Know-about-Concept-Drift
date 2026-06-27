import numpy as np

##TODO: This is all a bit dirty... FIXME

def random_pos(length, number, min_dist, min_dist_border):
    best_val, best = -1, None
    for _ in range(100):
        pos = list(np.random.randint(min_dist_border,length-min_dist_border,number))+[0,length]
        pos.sort()
        val = np.diff(pos).min()
        if val > best_val:
            best_val, best = val, np.array(pos)
            if best_val > min_dist:
                break
    return best

def gen_random(number=1, dims=5, intens=0.125, dist="unif", alt=False, length=750, min_dist=10, min_dist_border=100):
    pos = random_pos(length, number, min_dist, min_dist_border)
    e = np.zeros(length)
    for i,p in enumerate(pos[1:-1]):
        e[p:] += 1
    if alt:
        e %= 2
    
    if dist == "unif":
        y = np.zeros(length)
        y[pos[1:-1]] = intens if not alt else [intens*(-1)**i for i in range(pos.shape[0]-2)]
        X = np.random.random(size=(length,dims))
        X[:,:2] += np.cumsum(y)[:,None]
    elif dist == "gauss":
        assert alt
        X = np.random.normal(size=(length,dims))
        X[:,:2] += 6 * intens * X[:,:2].mean(axis=1)[:,None]
        for i,(p1,p2) in enumerate(zip(pos[:-1],pos[1:])):
            X[p1:p2,0] *= (-1)**i
    elif dist == "dubi":
        assert alt
        X = np.random.random(size=(length,dims))
        X[:int(length/2)] += 3*intens
        X -= X.mean(axis=0)[None,:]
        X = X[np.random.permutation(X.shape[0])]
        for i,(p1,p2) in enumerate(zip(pos[:-1],pos[1:])):
            X[p1:p2,0] *= (-1)**i
    else:
        raise ValueError("Distribution %s not defined"%dist)
    return X, e

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    X,y = gen_random(dist="unif")
    plt.scatter(X[:,0],X[:,1], c=y)
    plt.show()
