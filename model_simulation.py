from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from utils import T_HORIZON, GAMMA_SHAPE, GAMMA_SCALE

import numpy as np
import utils as uf

Label = Tuple[int, ...]
Path = np.ndarray


@dataclass
class EndState:
    t_end: float
    x_end: np.float64
    offspring_case: int
    t_end_ancestor: Optional[float]


@dataclass
class ParticleSample:
    label: Label
    t_start: float
    x_start: np.ndarray
    paths: Dict[Label, Path]
    end_states: List[EndState]
    t_end_ancestor: Optional[float] = None


def particles_generation(ctx: ParticleSample):
    """
    A local function that recursively simulates the movement of one particle.
    Returns:
        None. Updates `paths` in place. For the definition of `paths` see the comment for simulate_branching(...).
    """
    # Generation of the branching time
    tau = np.random.gamma(shape=GAMMA_SHAPE, scale=GAMMA_SCALE)
    t_end = min(ctx.t_start + tau, T_HORIZON)

    # Simulation of the particle's path
    X = uf.simulation_bm(t_start=ctx.t_start, t_end=t_end, x_start=ctx.x_start)
    ctx.paths[ctx.label] = X  # Store the path

    # Simulation of the number of descendant particles of the type 1 and the type 2
    I, I_prob_index = uf.sample_I()
    num_offspring_type_1, num_offspring_type_2 = I

    ctx.end_states.append(EndState(t_end, X[-1, 1], I_prob_index, ctx.t_end_ancestor))

    if t_end >= T_HORIZON:
        return  # Particle reached terminal time, no branching

    if num_offspring_type_1 == 0 and num_offspring_type_2 == 0:
        return

    child_index = 1

    # Recursive simulation of the descendant particle's paths
    for _ in range(num_offspring_type_1):
        new_label = ctx.label + (child_index,)
        particles_generation(ParticleSample(
            new_label,
            t_end, np.array([X[-1, -1]]),
            ctx.paths, ctx.end_states,
            t_end_ancestor=t_end))
        child_index += 1

    for _ in range(num_offspring_type_2):
        jump = uf.sample_jump(1)
        new_label = ctx.label + (child_index,)
        particles_generation(ParticleSample(
            new_label,
            t_end,
            np.array([X[-1, -1] + jump]),
            ctx.paths,
            ctx.end_states,
            t_end_ancestor=t_end))
        child_index += 1


def simulate_branching(
        label: Label,
        t_start: float,
        x_start: np.ndarray
):
    """
    A function to simulate the full branching process.
    Returns:
        dict: Keys are particle indices (tuple of ints),
        values are np.ndarray of shape (n, 2) — each row is (time point, position).
    """
    paths: Dict[Label, Path] = {}
    end_states: List[EndState] = []

    particles_generation(ParticleSample(label, t_start, x_start, paths, end_states))

    # Sort the dictionary by the particle indices for a better legend on the plot
    sorted_items = sorted(paths.items(), key=lambda item: (len(item[0]), item[0]))
    paths = dict(sorted_items)

    return paths, end_states


# Preparing the output for the further tasks: result[0] for plotting the branching process
#                                             result[1] for the MC Simulation
result = simulate_branching(label=(1,), t_start=0.0, x_start=np.array([[0.0]]))  # used further in plot.py
realization_for_plot = result[0]

sample_list = []
for i in range(1):
    sample_list.append(simulate_branching(label=(1,), t_start=0.0, x_start=np.array([[0.0]]))[1])

print(result)
