import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import svd, inv
from scipy.linalg import sqrtm
import numpy as np

def mppca_init_KSS(Y, J, k, max_iter=1000, n_init=10, random_state=0):
    d, N = Y.shape  # Original data dimension and number of samples

    # K-means++ initialization with more restarts
    kmeans = KMeans(n_clusters=J, init='k-means++', n_init=n_init, max_iter=max_iter, random_state=random_state)
    kmeans.fit(Y.T)
    mu0 = kmeans.cluster_centers_.T  # Initial means (d, J)
    assgn0 = kmeans.labels_  # Initial assignments (length N)

    # Check for empty clusters and reassign points if necessary
    for j in range(J):
        if np.sum(assgn0 == j) == 0:
            print(f"Cluster {j} is empty after K-means initialization.")
            random_idx = np.random.choice(N)
            assgn0[random_idx] = j
            mu0[:, j] = Y[:, random_idx]

    F0 = [np.zeros((d, k)) for _ in range(J)]  # Initial factor matrices
    U0 = [np.zeros((d, k)) for _ in range(J)]  # Initial bases

    # Run K-Subspaces algorithm
    KSS(Y, k, mu0, assgn0, F0, U0, max_iter)

    epsilon = 1e-6
    # Initialize mixing proportions and variances
    prop0 = np.array([np.sum(assgn0 == j) for j in range(J)], dtype=float)
    prop0 = np.maximum(prop0, epsilon)
    prop0 /= np.sum(prop0)  # Normalize so sum(prop0) = 1

    var0 = np.zeros(J)
    for j in range(J):
        idx = (assgn0 == j)
        n_j = np.sum(idx)
        if n_j <= k:
            print(f"Cluster {j} has insufficient points for variance estimation.")
            var0[j] = epsilon
            continue

        resid = Y[:, idx] - mu0[:, j:j+1]  # Residuals (d, n_j)
        cov_mat = (1 / n_j) * resid @ resid.T
        cov_mat = (cov_mat + cov_mat.T) / 2  # Ensure symmetry

        s = svd(cov_mat, compute_uv=False)
        if d - k > 0:
            var0[j] = (1 / (d - k)) * np.sum(s[k:])
            var0[j] = max(var0[j], epsilon)
        else:
            var0[j] = epsilon

    return F0, U0, mu0, prop0, var0

def KSS(Y, k, mu0, assgn0, F0, U0, num_iter):
    """
    K-Subspaces algorithm.

    Parameters:
    - Y: Data matrix (d, N).
    - k: Subspace dimension.
    - mu0: Initial means (d, J).
    - assgn0: Initial assignments (length N).
    - F0: List of factor matrices.
    - U0: List of bases.
    - num_iter: Number of iterations.
    """
    for i in range(num_iter):
        update_F_KSS(Y, k, mu0, assgn0, F0, U0)
        update_assgn_KSS(Y, mu0, assgn0, U0)
        update_mu_KSS(Y, mu0, assgn0)

def update_F_KSS(Y, k, mu0, assgn0, F0, U0):
    """
    Update factor matrices and bases.

    Parameters:
    - Y: Data matrix (d, N).
    - k: Subspace dimension.
    - mu0: Means (d, J).
    - assgn0: Assignments (length N).
    - F0: List of factor matrices.
    - U0: List of bases.
    """
    J = len(F0)
    d, N = Y.shape
    for j in range(J):
        idx = (assgn0 == j)
        n_j = np.sum(idx)
        if n_j < 0.1 * N:
            # Borrow samples from other clusters
            resid = Y - mu0[:, j:j+1]
            errs = np.sum(resid ** 2, axis=0)
            min_errs_idx = np.argsort(errs)[:int(round(0.1 * N))]
            idx[min_errs_idx] = True
            n_j = np.sum(idx)

        resid = Y[:, idx] - mu0[:, j:j+1]
        cov_mat = (1 / n_j) * resid @ resid.T

        U, s, _ = svd(cov_mat)
        sigma2 = (1 / max(d - k, 1e-6)) * np.sum(s[k:])
        sqrt_vals = np.sqrt(np.maximum(s[:k] - sigma2, 0))
        F0[j] = U[:, :k] @ np.diag(sqrt_vals)
        U0[j] = U[:, :k]

def update_assgn_KSS(Y, mu0, assgn0, U0):
    """
    Update cluster assignments.

    Parameters:
    - Y: Data matrix (d, N).
    - mu0: Means (d, J).
    - assgn0: Assignments (length N).
    - U0: List of bases.
    """
    N = Y.shape[1]
    J = mu0.shape[1]
    best = np.full(N, np.inf)
    for j in range(J):
        U = U0[j]
        resid = Y - mu0[:, j:j+1]
        # Projection error onto orthogonal complement of U
        proj_error = resid - U @ (U.T @ resid)
        errors = np.sum(proj_error ** 2, axis=0)
        better = errors < best
        assgn0[better] = j
        best[better] = errors[better]

def update_mu_KSS(Y, mu0, assgn0):
    """
    Update cluster means.

    Parameters:
    - Y: Data matrix (d, N).
    - mu0: Means (d, J).
    - assgn0: Assignments (length N).
    """
    J = mu0.shape[1]
    for j in range(J):
        idx = (assgn0 == j)
        if np.sum(idx) == 0:
            continue  # Avoid division by zero
        mu0[:, j] = np.mean(Y[:, idx], axis=1)