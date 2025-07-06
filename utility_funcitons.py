import numpy as np
import matplotlib.pyplot as plt


def bm_simulation(d, T, n_step_per_1):
    """
    Simulate d-dimensional standard Brownian motion over [0, T].

    Returns:
        W: ndarray of shape (d, n_steps), each row is one BM path
    """
    n_steps = int(n_step_per_1 * T)
    dt = T / n_steps

    # Simulate independent Brownian motions
    increments = np.random.normal(loc=0, scale=np.sqrt(dt), size=(d, n_steps-1))
    W = np.cumsum(increments, axis=1)
    W = np.hstack([np.zeros((d, 1)), W])  # Add initial zero at time 0

    return W


def gbm_simulation(d, T, X_0, mu, C, n_step_per_1=1000):
    """
    Simulate d-dimensional Geometric Brownian Motion over [0, T].

    Args:
        d (int): number of assets
        T (float): total time
        X_0 (ndarray): initial values, shape (d,)
        mu (ndarray): drift vector, shape (d,)
        C (ndarray): covariance matrix, shape (d, d)
        n_step_per_1 (int): time discretization resolution

    Returns:
        X: ndarray of shape (d, n_steps), GBM paths
    """
    n_steps = int(n_step_per_1 * T)
    t = np.linspace(0, T, n_steps)

    # Cholesky decomposition for correlation structure
    L = np.linalg.cholesky(C)

    # Simulate independent Brownian motions
    W = bm_simulation(d, T, n_step_per_1)  # shape: (d, n_steps)

    # Correlated Brownian motions
    W_correlated = L @ W  # shape: (d, n_steps)

    # Drift adjustment: μ - ½ * Var (elementwise)
    drift = (mu - 0.5 * np.diag(C))[:, None]  # shape: (d, 1)

    # Full exponent for GBM
    exponent = drift * t + W_correlated  # shape: (d, n_steps)

    # Element-wise multiplication of initial values
    X = X_0[:, None] * np.exp(exponent)  # shape: (d, n_steps)

    return X
