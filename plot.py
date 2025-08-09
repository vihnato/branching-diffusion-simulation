from typing import Dict, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model_simulation import realization_for_plot

matplotlib.use('TkAgg')  # Solution for the error: 'FigureCanvasInterAgg' object has no attribute 'tostring_rgb'


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
