from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils as uf

T = 1  # Final simulation time (can be made a parameter if needed)


def particles_generation(
        label: Tuple[int, ...],
        t_start: float,
        x_start: np.ndarray,
        paths: Dict[Tuple[int, ...], np.ndarray]
):
    """
    Recursive helper that simulates one particle and its descendants.
    """
    tau = np.random.gamma(shape=0.5, scale=2.5)
    t_end = min(t_start + tau, T)

    X = uf.simulation_bm(d=1, t_start=t_start, t_end=t_end, x_start=x_start)
    paths[label] = X  # Store trajectory

    if t_end >= T:
        return  # Particle reached terminal time, no branching

    I, prob = uf.sample_I()
    num_offspring_type_1, num_offspring_type_2 = I
    child_index = 1

    for _ in range(num_offspring_type_1):
        new_label = label + (child_index,)
        particles_generation(new_label, t_end, np.array([X[-1, -1]]), paths)
        child_index += 1

    for _ in range(num_offspring_type_2):
        jump = uf.sample_jump(1)
        new_label = label + (child_index,)
        particles_generation(new_label, t_end, np.array([X[-1, -1] + jump]), paths)
        child_index += 1


def simulate_branching(
        label: Tuple[int, ...],
        t_start: float,
        x_start: np.ndarray
) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    A function to simulate the full branching process.
    """
    paths: Dict[Tuple[int, ...], np.ndarray] = {}
    particles_generation(label, t_start, x_start, paths)

    sorted_items = sorted(paths.items(), key=lambda item: (len(item[0]), item[0]))

    paths = dict(sorted_items)

    return paths


def plot_branching_paths(paths: Dict[Tuple[int, ...], np.ndarray]) -> None:
    """
    Plot all particle paths from a branching process.

    Args:
        paths: Dictionary where
            - keys are particle labels (e.g. (1,), (1,1), ...)
            - values are lists of (time, position) tuples
    """

    plt.figure(figsize=(15, 6))

    for label, path in paths.items():
        # print(label, path)

        times = path[:, 0]
        positions = path[:, 1]
        plt.plot(times, positions, label=str(label))
        plt.legend()


if __name__ == "__main__":
    for i in range(10):
        print("Starting simulation...")

        seed = np.random.randint(0, 10 ** 6)  # generate a random seed
        np.random.seed(seed)

        paths = simulate_branching(label=(1,), t_start=0.0, x_start=np.array([[0.0]]))
        plot_branching_paths(paths)
        plt.title(f"Seed used: {seed}")
        plt.show()
