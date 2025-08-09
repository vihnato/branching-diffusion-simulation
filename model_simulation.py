from typing import Dict, Tuple, Optional

import numpy as np
import utils as uf

T = 1  # Final simulation time (can be made a parameter if needed)


def particles_generation(
        label: Tuple[int, ...],
        t_start: float,
        x_start: np.ndarray,
        paths: Dict[Tuple[int, ...], np.ndarray],
        end_state: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, bool, Optional[int]]],
        offspring_case: Optional[int] = None
):
    """
    A local function that recursively simulates the movement of one particle.
    Returns:
        None. Updates `paths` in place. For the definition of `paths` see the comment for simulate_branching(...).
    """
    # Generation of the branching time
    tau = np.random.gamma(shape=0.5, scale=2.5)
    t_end = min(t_start + tau, T)

    # Simulation of the particle's path
    X = uf.simulation_bm(t_start=t_start, t_end=t_end, x_start=x_start)
    paths[label] = X  # Store the path

    if t_end >= T:
        end_state[label] = (X[-1, 0], X[-1, 1], True, offspring_case)
        return  # Particle reached terminal time, no branching

    # Simulation of the number of descendant particles of the type 1 and the type 2
    I, prob, i = uf.sample_I()
    num_offspring_type_1, num_offspring_type_2 = I
    child_index = 1

    if num_offspring_type_1 == 0 and num_offspring_type_2 == 0:
        end_state[label] = (X[-1, 0], X[-1, 1], False, offspring_case)
        return

    # Recursive simulation of the descendant particle's paths
    for _ in range(num_offspring_type_1):
        new_label = label + (child_index,)
        particles_generation(new_label, t_end, np.array([X[-1, -1]]), paths, end_state, i)
        child_index += 1

    for _ in range(num_offspring_type_2):
        jump = uf.sample_jump(1)
        new_label = label + (child_index,)
        particles_generation(new_label, t_end, np.array([X[-1, -1] + jump]), paths, end_state, i)
        child_index += 1


def simulate_branching(
        label: Tuple[int, ...],
        t_start: float,
        x_start: np.ndarray
) -> Tuple[
        Dict[Tuple[int, ...], np.ndarray],
        Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, bool, Optional[int]]]
        ]:
    """
    A function to simulate the full branching process.
    Returns:
        dict: Keys are particle indices (tuple of ints),
        values are np.ndarray of shape (n, 2) — each row is (time point, position).
    """
    paths: Dict[Tuple[int, ...], np.ndarray] = {}
    end_state: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, bool, Optional[int]]] = {}

    particles_generation(label, t_start, x_start, paths, end_state)

    # Sort the dictionary by the particle indices for a better legend on the plot
    sorted_items = sorted(paths.items(), key=lambda item: (len(item[0]), item[0]))
    paths = dict(sorted_items)

    return paths, end_state


result = simulate_branching(label=(1,), t_start=0.0, x_start=np.array([[0.0]]))  # used further in plot.py
realization_for_plot = result[0]
print(result[1])
