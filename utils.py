import numpy as np

probs = [1/3, 1/6, 1/6, 1/6, 1/12, 1/12]
T_HORIZON = 1
ALPHA = 0.2
GAMMA_SHAPE = 0.5
GAMMA_SCALE = 2.5


def simulation_bm(
    t_start: float,
    t_end: float,
    x_start: np.ndarray = np.array([0.0]),
    steps_per_unit: int = 10000
) -> np.ndarray:
    """
    Simulate d-dimensional standard Brownian motion over [t_start, t_end],
    returning time-position pairs for d = 1.

    Returns:
        np.ndarray of shape (n_steps, 2), where each row is [time, position]
    """
    n_steps = max(1, int(steps_per_unit * (t_end - t_start)))
    dt = (t_end - t_start) / n_steps

    # Simulate 1D Brownian motion
    increments = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=n_steps-1)
    W = np.cumsum(increments)
    W += x_start[0]
    W = np.insert(W, 0, x_start[0])  # include starting point

    time_points = np.linspace(t_start, t_end, n_steps)

    return np.column_stack((time_points, W))

X = simulation_bm(0, 1)[0]
print(type(X))


# TODO: get back to this function later
def simulation_gbm(
    d: int,
    t_start: float,
    t_end: np.ndarray,
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


def sample_I() -> tuple[np.ndarray, int]:
    values = [
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([2, 0]),
        np.array([0, 2])
    ]

    i = np.random.choice(len(values), p=probs)

    return values[i], i


def sample_jump(d):
    i = np.random.randint(d)    # Choose i ∈ {0, ..., d-1}
    jump = np.zeros(d)
    jump[i] = -np.pi / 2        # Set -π/2 in the i-th coordinate
    return jump
