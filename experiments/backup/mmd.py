import numpy as np

# from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
# from mmd_variants import rbf_kernel as apply_kernel, HAS_TORCH, DEVICE
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
from mmd_variants import HAS_TORCH, DEVICE

# if HAS_TORCH:
#     import torch


def gen_window_matrix(l1, l2, n_perm, cache=dict()):
    # Convert tuple key to string or robust key if needed, but tuple works for dict
    if (l1, l2, n_perm) not in cache.keys():
        w = np.array(l1 * [1.0 / l1] + (l2) * [-1.0 / (l2)])
        W = np.array([w] + [np.random.permutation(w) for _ in range(n_perm)])

        # if HAS_TORCH:
        #     # Pre-convert to tensor if using GPU
        #     cache[(l1, l2, n_perm)] = torch.tensor(
        #         W, device=DEVICE, dtype=torch.float32
        #     )
        # else:
        cache[(l1, l2, n_perm)] = W

    return cache[(l1, l2, n_perm)]


def mmd(X, s=None, n_perm=2500):
    # X can be numpy or torch
    # K = apply_kernel(X, X, gamma="auto")  # rbf_kernel handles torch/numpy logic
    K = apply_kernel(X, metric="rbf")  # Reverted to sklearn default

    if s is None:
        s = int(X.shape[0] / 2)

    W = gen_window_matrix(s, K.shape[0] - s, n_perm)

    # if HAS_TORCH and isinstance(K, torch.Tensor):
    if False:  # GPU branch disabled
        pass
    else:
        # Numpy path
        W_np = W
        # if HAS_TORCH and isinstance(W, torch.Tensor):
        #     W_np = W.cpu().numpy()

        WK = np.dot(W_np, K)
        s_stat = np.einsum("ij,ij->i", WK, W_np)
        p = (s_stat[0] < s_stat).sum() / n_perm

        return s_stat[0], p
