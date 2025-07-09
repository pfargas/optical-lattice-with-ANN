from collections.abc import Sequence
import numpy as np
import warnings
from constants import Constants
from numpy import ndarray
import itertools


def load_constants():
    for name, value in Constants.__members__.items():
        globals()[name] = value.value


def chain_to_idx(input, N_states, N_particles):
    """Compute the index for a given input state."""
    if isinstance(input, int):
        return input
    if np.any(np.array(input) >= N_states):
        raise ValueError("Input indices must be less than N_states.")
    elif isinstance(input, Sequence) and len(input) == N_particles:
        idx = 0
        for i in range(1, N_particles + 1):
            idx += input[i - 1] * (N_states ** (N_particles - i))
        return idx
    elif isinstance(input, int):
        pass
    else:
        print(type(input))
        raise ValueError("Input must be an integer or an iterable of particle indices.")


def idx_to_chain(input, N_states, N_particles):

    array = []
    for particle in range(N_particles):
        array.append(np.floor_divide(input, N_states ** (N_particles - particle - 1)))
        input = input - array[-1] * N_states ** (N_particles - particle - 1)
    return np.array(array, dtype=int)


class SpaceDiscretization:
    """Class for discretizing space in a quantum system.

    Attributes:
        N_points (int): Number of discretized points in space.
        L_box (float): Length of the box in which the system is defined.
        bounds (tuple): Bounds of the system, if provided.
        dx (float): Discretization step size.

    NOTE: Either L_box or bounds must be provided, if both provided, L_box will be used.
    L_box is defined such that the system is in a box of length L_box centered at zero.
    That means that bounds = (-L_box/2, L_box/2) if bounds is not provided.
    """

    def __init__(self, N_points=None, L_box=None, bounds=None, dx=None, PBC=True):
        self.PBC = PBC  # Periodic boundary conditions
        if L_box is None and bounds is None:
            raise ValueError("Either L_box or bounds must be provided.")
        if L_box is not None and bounds is not None:
            warnings.warn("Both L_box and bounds provided, using L_box.")
        if L_box is not None:
            self.L_box = L_box
            self.bounds = (-L_box / 2, L_box / 2)
        else:
            self.bounds = bounds
            self.L_box = bounds[1] - bounds[0]

        if N_points is None and dx is None:
            raise ValueError("Either N_points or dx must be provided.")
        if N_points is not None and dx is not None:
            warnings.warn("Both N_points and dx provided, using N_points.")
        if N_points is not None:
            self.N_points = N_points
            self.dx = self.L_box / N_points
        else:
            self.dx = dx
            self.N_points = int(self.L_box / dx)
