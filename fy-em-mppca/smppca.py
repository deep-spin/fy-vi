from entmax.activations import sparsemax, entmax15
from entmax.root_finding import entmax_bisect
import torch

import numpy as np
from scipy.linalg import inv, pinv
from scipy.special import logsumexp


def smppca_tipping_final(Y, F0, mu0, prop0, var0, niter=1000, eps=1e-6, epsilon=1e-8, T = 2, alpha=1, anneal=False):
    """
    homppca_tipping_final(Y, prop0, F0, var0, niter): function that implements homoscedastic MPPCA

    Constants:
        - N: number of total data samples
        - d: data dimension
        - k: intrinsic dimension (k < d)
        - J: number of mixtures

    INPUTS:
        - Y: data samples (size (d, N) matrix, where each column is a sample)
        - F0: initial guess on factor matrices (list of (d, k) numpy arrays of length J)
        - mu0: initial guess on mixture means (size (d, J) numpy array, where each column is a mean)
        - prop0: initial guess on mixing proportions (numpy array of length J)
        - var0: initial guess on variance (numpy array of length J)
        - niter: maximum number of EM iterations to run (scalar)
        - eps: convergence threshold for EM iterations
        - epsilon: small constant to prevent numerical issues

    RETURNS:
        - F: factor matrix estimates
        - mu: mixture means estimates
        - var: noise variance estimates
        - prop: mixing proportions
    """
    # Define constants
    N = Y.shape[1]  # Number of data samples
    d, k = F0[0].shape  # Original dimension (d), intrinsic dimension (k)
    # print("d", d, "k", k)
    J = len(prop0)  # Number of mixtures
    
    # Parameters to estimate
    F = [np.copy(f) for f in F0]  # Factor matrices
    mu = np.copy(mu0)  # Mixture means
    prop = np.copy(prop0)  # Mixing proportions
    var = np.copy(var0)  # Noise variances

    # Add epsilon to variances and proportions to prevent zeros
    var = np.maximum(var, epsilon)

    # try this after removing prop thing
    prop = np.maximum(prop, epsilon)
    prop /= np.sum(prop)  # Re-normalize proportions

    # Useful variables
    M = [np.zeros((k, k)) for _ in range(J)]  # Posterior covariances
    M_inv = [np.zeros((k, k)) for _ in range(J)]  # Inverses of posterior covariances
    C_inv = [np.zeros((d, d)) for _ in range(J)]  # Inverses of marginal covariances
    R = np.zeros((N, J))  # Log responsibilities

    # Perform EM iterations
    
    for itr in range(niter):
        # Expectation step
        # print(itr+1)
        # T = 1/np.log(itr+2)
        
        s_E_step_tipping_final(M, M_inv, C_inv, R, Y, F, mu, prop, var, T, alpha, epsilon)
        if anneal:
            alpha += 0.001
            
        # Maximization step
        F_prev = [np.copy(f) for f in F]
        s_M_step_tipping_final(prop, mu, F, var, Y, R, M_inv, epsilon)

        # Termination criteria
        norm_diffs = [np.linalg.norm(F_prev[i] - F[i]) / (np.linalg.norm(F_prev[i]) + epsilon) for i in range(J)]
        if max(norm_diffs) < eps:
            break
        
    return F, mu, var, prop

def s_E_step_tipping_final(M, M_inv, C_inv, R, Y, F, mu, prop, var, T, alpha, epsilon=1e-8):
    """
    E-step of EM algorithm in non-log scale for homoscedastic MPPCA.
    
    Parameters:
        M, M_inv, C_inv: Lists to store covariance matrices and their inverses.
        R: Responsibilities matrix to be updated (N, J).
        Y: Data matrix (d, N).
        F: List of factor matrices (each d x k).
        mu: Mixture means matrix (d, J).
        prop: Mixing proportions array (J,).
        var: Noise variances array (J,).
        epsilon: Small constant to prevent numerical issues.
    """
    N = Y.shape[1]
    d = Y.shape[0]
    k = F[0].shape[1]
    J = len(prop)
    
    # Initialize exponents matrix
    log_R = np.copy(R)
    
    for j in range(J):
        # 1. Compute M_j and M_j_inv
        M[j] = var[j] * np.eye(k) + F[j].T @ F[j]
        try:
            M_inv[j] = inv(M[j])
        except np.linalg.LinAlgError:
            M_inv[j] = pinv(M[j])
        
        # 2. Compute C_inv_j
        C_inv_j = (1.0 / var[j]) * (np.eye(d) - F[j] @ M_inv[j] @ F[j].T)
        C_inv[j] = C_inv_j
        
        # 3. Compute log determinant of C_inv_j
        sign, logdet_C_inv_j = np.linalg.slogdet(C_inv_j)
        if sign <= 0 or not np.isfinite(logdet_C_inv_j):
            logdet_C_inv_j = np.log(epsilon)
        
        
        tmp = -0.5 * np.sum((Y - mu[:, [j]]) * (C_inv_j @ (Y - mu[:, [j]])), axis=0)
        
        # 5. Compute exponents including constants
        prop_j = max(prop[j], epsilon)
        log_R[:, j] = 0.5 * (logdet_C_inv_j) + tmp  - (d / 2) * np.log(2 * np.pi) + np.log(prop_j)  # ln pi_i * p(t_n|i)
        # log_R[:, j] = 0.5 * (logdet_C_inv_j) + tmp  - (d / 2) * np.log(2 * np.pi) + prop_j  # ln pi_i * p(t_n|i)


    # ln_sum (over classes)
    max_log_R = np.max(log_R, axis=1, keepdims=True)  # Shape: (N, J)
    unnormalized_R = log_R - max_log_R # Shape: (N, J)
    # R[:, :] = entmax_bisect(1/T*torch.tensor(unnormalized_R), alpha=alpha).numpy()  # Broadcasting division
    R[:, :] = entmax_bisect(torch.tensor(np.exp(unnormalized_R)), alpha=alpha).numpy()   # Broadcasting division
    # print("after", R[:2])


def s_M_step_tipping_final(prop, mu, F, var, Y, R, M_inv, epsilon):
    """
    Function that performs maximization step of EM
    """
    s_update_prop_tipping_final(prop, R, epsilon)
    s_update_mu_tipping_final(mu, Y, F, R, M_inv, epsilon)
    s_update_F_tipping_final(F, Y, mu, R, var, M_inv, epsilon)
    s_update_var_tipping_final(var, F, mu, Y, R, M_inv, epsilon)

def s_update_prop_tipping_final(prop, R, epsilon=1e-8):
    N = R.shape[0]
    prop[:] = np.sum(R, axis=0) / N
    prop = np.maximum(prop, epsilon)  # Ensuring no zero proportions
    prop /= np.sum(prop)  # Normalize to ensure it sums to 1

def s_update_mu_tipping_final(mu, Y, F, R, M_inv, epsilon=1e-8):
    """
    Update mixture means with checks to prevent division by zero.
    """
    N = Y.shape[1]
    J = len(F)
    d, k = F[0].shape

    mu_num = np.zeros((d, J))
    mu_den = np.zeros(J) + epsilon  # Prevent zero denominator

    for j in range(J):
        R_j = R[:, j]  # Responsibilities for mixture j
        for i in range(N):
            y_i = Y[:, i]
            resid = y_i - mu[:, j]
            z_ij = M_inv[j] @ (F[j].T @ resid)
            mu_num[:, j] += R_j[i] * (y_i - F[j] @ z_ij)
        mu_den[j] = np.sum(R_j) + epsilon

    mu[:, :] = mu_num / mu_den[np.newaxis, :]

def s_update_F_tipping_final(F, Y, mu, R, var, M_inv, epsilon=1e-8):
    N = Y.shape[1]
    J = len(F)
    d, k = F[0].shape

    for j in range(J):
        Fj1_new = np.zeros((d, k))
        Fj2_new = np.zeros((k, k))

        for i in range(N):
            y_i = Y[:, i]
            resid = y_i - mu[:, j]
            z_ij = M_inv[j] @ (F[j].T @ resid)
            Z_ij = var[j] * M_inv[j] + np.outer(z_ij, z_ij)

            weight = R[i, j]

            Fj1_new += weight * np.outer(resid, z_ij)
            Fj2_new += weight * Z_ij

        # Regularize Fj2_new
        reg_param = epsilon * np.trace(Fj2_new) / k  # Scale regularization
        Fj2_new += reg_param * np.eye(k)

        # Use pseudoinverse if necessary
        try:
            F_inv = inv(Fj2_new)
        except np.linalg.LinAlgError:
            F_inv = pinv(Fj2_new)

        F[j][:, :] = Fj1_new @ F_inv

def s_update_var_tipping_final(var, F, mu, Y, R, M_inv, epsilon=1e-8):
    """
    Update variances with checks to prevent zero or negative values.
    """
    N = Y.shape[1]
    J = len(F)
    d = F[0].shape[0]

    vnew_num = np.zeros(J)
    vnew_den = np.zeros(J) + epsilon  # Prevent zero denominator

    for j in range(J):
        R_j = R[:, j]
        for i in range(N):
            y_i = Y[:, i]
            resid = y_i - mu[:, j]
            z_ij = M_inv[j] @ (F[j].T @ resid)
            Z_ij = var[j] * M_inv[j] + np.outer(z_ij, z_ij)
            term1 = np.linalg.norm(resid) ** 2
            term2 = -2 * z_ij.T @ (F[j].T @ resid)
            term3 = np.trace((F[j].T @ F[j]) @ Z_ij)
            vnew_num[j] += R_j[i] * (term1 + term2 + term3)
        vnew_den[j] = d * np.sum(R_j) + epsilon

    var[:] = vnew_num / vnew_den
    var = np.maximum(var, epsilon)  # Ensure variances are positive