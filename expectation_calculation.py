import numpy as np

from model_simulation import sample_list
from scipy.stats import gamma

from utils import T_HORIZON, ALPHA, GAMMA_SHAPE, GAMMA_SCALE


def c_function(index, t_end, x_end, alpha=ALPHA, T=T_HORIZON, d=1):
    """
    Compute one of the c_{i,j}(t, x) formulas by index:
      0 -> c_(0,0)
      1 -> c_(1,0)
      2 -> c_(0,1)
      3 -> c_(2,0)
      4 -> c_(0,2)
      5 -> c_(1,1)

    Parameters
    ----------
    index : int (0-5)
        Formula selector.
    t_end : float
        Time t.
    x_end : np.ndarray
        Vector x (shape: (d,))
    alpha : float
        Alpha parameter.
    d : int
        Dimension.
    T : float
        Final time.
    """
    one_dot_x = np.sum(x_end)  # 1_d^T x

    if index == 0:
        return ((alpha + 1 / (2 * d)) * np.cos(one_dot_x) * np.exp(alpha * (T - t_end))
                + (np.cos(one_dot_x) ** 2) / d - 1 / (2 * d))

    elif index == 1:
        return (-1) * np.cos(one_dot_x) * np.exp(-alpha * (T - t_end)) / d

    elif index == 2:
        return (-1) * c_function(1, t_end, x_end, alpha, T, d)

    elif index == 3:
        return np.exp(-2 * alpha * (T - t_end)) / (2 * d)

    elif index == 4:
        return c_function(3, t_end, x_end, alpha, T, d)

    elif index == 5:
        return (-2) * c_function(3, t_end, x_end, alpha, T, d)

    else:
        print("Index must be in [0, 5]")
        return None


def survival(t):
    """
    Survival function for Gamma(shape=0.5, scale=2.5).

    S(t) = 1 - F(t), where F is the CDF of the gamma distribution.
    """
    return 1.0 - gamma.cdf(t, a=GAMMA_SHAPE, scale=GAMMA_SCALE)


for realization in sample_list:
    survived = 1
    dead = 1
    for particle in realization:
        if particle.t_end == 1:
            payout_val = np.cos(particle.x_end)
            if particle.t_end_ancestor is None:
                survived *= payout_val / survival(1)
            else:
                survived *= np.cos(particle.x_end) / survival(1 - particle.t_end_ancestor)
        else:
            payout_val = c_function(index=particle.offspring_case, t_end=particle.t_end, x_end=particle.x_end)
            if particle.t_end_ancestor is None:
                cdf = gamma.cdf(particle.t_end, a=GAMMA_SHAPE, scale=GAMMA_SCALE)
                dead *= payout_val / cdf
            else:
                cdf = gamma.cdf(particle.t_end - particle.t_end_ancestor, a=GAMMA_SHAPE, scale=GAMMA_SHAPE)
                dead *= payout_val / cdf

    print(survived, dead)


from typing import Dict, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model_simulation import realization_for_plot

# matplotlib.use('TkAgg')  # Solution for the error: 'FigureCanvasInterAgg' object has no attribute 'tostring_rgb'


def plot_branching_paths(paths: Dict[Tuple[int, ...], np.ndarray]) -> None:
    """
    Plot all particle paths from a diffusion branching process.
    """

    plt.figure(figsize=(15, 6))

    for label, path in paths.items():
        # print(label, path)

        times = path[:, 0]
        positions = path[:, 1]
        plt.plot(times, positions, label=str(label))
        plt.legend()


if __name__ == "__main__":
    seed = np.random.randint(0, 10 ** 6)  # generate a random seed
    np.random.seed(seed)

    plot_branching_paths(realization_for_plot)
    plt.title(f"Seed used: {seed}")
    plt.show()
