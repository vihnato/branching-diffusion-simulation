import numpy as np
import matplotlib.pyplot as plt


def simulation_bm(
    d: int,
    t_start: float,
    t_end: float,
    x_start: np.ndarray = np.array([0.0]),
    steps_per_unit: int = 10000
) -> np.ndarray:
    """
    Simulate d-dimensional standard Brownian motion over [t_start, t_end],
    returning time-position pairs for d = 1.

    Returns:
        np.ndarray of shape (n_steps + 1, 2), where each row is [time, position]
    """
    assert d == 1, "This function only supports d = 1 for time-position output"

    n_steps = max(1, int(steps_per_unit * (t_end - t_start)))
    dt = (t_end - t_start) / n_steps

    # Simulate 1D Brownian motion
    increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps)
    W = np.cumsum(increments)
    W += x_start[0]
    W = np.insert(W, 0, x_start[0])  # include starting point

    times = np.linspace(t_start, t_end, n_steps + 1)

    return np.column_stack((times, W))


def simulation_gbm(
    d: int,
    t_start: float,
    t_end: float,
    x_start: np.ndarray,  # shape (d,)
    mu: np.ndarray,       # shape (d,)
    C: np.ndarray,        # shape (d, d)
    n_step_per_1: int = 1000
) -> np.ndarray:
    """
    Simulate d-dimensional Geometric Brownian Motion over [t_start, t_end].
    """
    # Ensure inputs are at least 1D/2D
    x_start = np.atleast_1d(x_start)
    mu = np.atleast_1d(mu)
    C = np.atleast_2d(C)

    n_steps = int(n_step_per_1 * (t_end - t_start))
    t = np.linspace(t_start, t_end, n_steps + 1)

    # Brownian motion
    W = simulation_bm(d, t_start, t_end, n_step_per_1)  # shape: (d, n_steps + 1)

    # Drift and correlation
    if d == 1:
        drift = np.array([[mu[0] - 0.5 * C[0, 0]]])  # shape: (1,1)
        W_correlated = W
    else:
        L = np.linalg.cholesky(C)
        W_correlated = L @ W  # shape: (d, n_steps + 1)
        drift = (mu - 0.5 * np.diag(C))[:, None]  # shape: (d,1)

    exponent = drift * t + W_correlated  # broadcasting over time steps
    x = x_start[:, None] * np.exp(exponent)  # shape: (d, n_steps + 1)

    return x


def sample_I() -> tuple[np.ndarray, float]:
    values = [
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([2, 0]),
        np.array([0, 2])
    ]
    probs = [1/3, 1/6, 1/6, 1/6, 1/12, 1/12]

    i = np.random.choice(len(values), p=probs)

    return values[i], probs[i]


def sample_jump(d):
    i = np.random.randint(d)    # Choose i ∈ {0, ..., d-1}
    jump = np.zeros(d)
    jump[i] = -np.pi / 2        # Set -π/2 in the i-th coordinate
    return jump
