from collections.abc import Sequence
import numpy as np
import warnings
from constants import Constants
from numpy import ndarray
import itertools


def load_constants():
    for name, value in Constants.__members__.items():
        globals()[name] = value.value


load_constants()


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
    else:
        raise ValueError("Input must be an integer or an iterable of particle indices.")


def idx_to_chain(input, N_states, N_particles):

    array = []
    for particle in range(N_particles):
        array.append(np.floor_divide(input, N_states ** (N_particles - particle - 1)))
        input = input - array[-1] * N_states ** (N_particles - particle - 1)
    return np.array(array, dtype=int)


class DiscreteState:
    def __init__(self, N_states, N_particles, initial_state=None, initial_idx=None):
        # N_states and N_particles must be inmutable
        self._N_states = N_states
        self._N_particles = N_particles
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=int)
        elif initial_idx is not None:
            self.state = idx_to_chain(initial_idx, N_states, N_particles)
        elif initial_idx is None and initial_state is None:
            warnings.warn("No initial state provided, initializing to zeros.")
            self.state = np.zeros(N_particles, dtype=int)

    def flat_idx(self):
        """Return the flat index of the current state."""
        return chain_to_idx(self.state, self._N_states, self._N_particles)

    def idx_to_state(self, idx):
        """Convert a flat index to a state."""
        return idx_to_chain(idx, self._N_states, self._N_particles)

    @property
    def N_states(self):
        return self._N_states

    @property
    def N_particles(self):
        return self._N_particles

    def __repr__(self):

        string_state = "|"
        for s in self.state:
            string_state += f"{s},"
        string_state = string_state[:-1]
        string_state += ">"

        return f"DiscreteState(N_states={self.N_states}, N_particles={self.N_particles}, state={string_state})"


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


def kinetic_matrix_one_particle(space_properties: SpaceDiscretization = None):
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    PBC = space_properties.PBC
    dx = space_properties.dx
    N_points = space_properties.N_points

    kin_coeff = -0.5 * HBAR**2 / (MASS * dx**2)

    kin_matrix = np.zeros((N_points, N_points))
    main_diag = -2 * np.ones(N_points)
    off_diag = np.ones(N_points - 1)
    np.fill_diagonal(kin_matrix, main_diag)
    np.fill_diagonal(kin_matrix[1:], off_diag)
    np.fill_diagonal(kin_matrix[:, 1:], off_diag)
    if PBC:
        kin_matrix[0, -1] = 1
        kin_matrix[-1, 0] = 1
    kin_matrix *= kin_coeff
    return kin_matrix


def kinetic_matrix_many_particles(
    space_properties: SpaceDiscretization = None, N_particles=1
):
    """Construct the kinetic matrix for a many-particle system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)
    N_points = space_properties.N_points

    II = np.eye(space_properties.N_points)
    kin_matrix_one_p = kinetic_matrix_one_particle(space_properties)
    kin_matrix_many_body = np.zeros((N_points**N_particles, N_points**N_particles))

    for i in range(N_particles):
        factors = []
        for p in range(N_particles):
            if p == i:
                factors.append(kin_matrix_one_p)
            else:
                factors.append(II)
        print(f"Term {i+1}:")
        for factor in factors:
            print(f"\n{factor}\n")
        print("\n________________________\n")

        # Kronecker product of all factors
        term = factors[0]
        for f in factors[1:]:
            term = np.kron(term, f)

        kin_matrix_many_body += term
    return kin_matrix_many_body


def one_body_operator_many_particles(
    operator: ndarray, space_properties: SpaceDiscretization = None, N_particles=1
):
    """Construct the operator for a many-particle system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    N_points = space_properties.N_points
    II = np.eye(N_points)
    operator_one_p = operator
    operator_many_body = np.zeros((N_points**N_particles, N_points**N_particles))

    for i in range(N_particles):
        factors = []
        for p in range(N_particles):
            if p == i:
                factors.append(operator_one_p)
            else:
                factors.append(II)

        # Kronecker product of all factors
        term = factors[0]
        for f in factors[1:]:
            term = np.kron(term, f)

        operator_many_body += term

    return operator_many_body


def kinetic_matrix_many_particles(
    space_properties: SpaceDiscretization = None, N_particles=1
):
    """Construct the kinetic matrix for a many-particle system."""
    kin_matrix_one_p = kinetic_matrix_one_particle(space_properties)
    return one_body_operator_many_particles(
        kin_matrix_one_p, space_properties=space_properties, N_particles=N_particles
    )


def potential_matrix_one_body(
    V: callable, space_properties: SpaceDiscretization = None, *args, **kwargs
):
    """Construct the potential matrix for a one-body system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    N_points = space_properties.N_points
    dx = space_properties.dx
    x = np.linspace(space_properties.bounds[0], space_properties.bounds[1], N_points)

    pot_matrix = np.zeros((N_points, N_points))
    for i in range(N_points):
        pot_matrix[i, i] = V(x[i], *args, **kwargs)

    return pot_matrix


def potential_matrix_many_body(
    V: callable,
    space_properties: SpaceDiscretization = None,
    N_particles=1,
    *args,
    **kwargs,
):
    """Construct the potential matrix for a many-body system."""
    pot_matrix_one_p = potential_matrix_one_body(V, space_properties, *args, **kwargs)
    return one_body_operator_many_particles(
        pot_matrix_one_p, space_properties=space_properties, N_particles=N_particles
    )


def interaction_matrix_two_body(
    U: callable,
    space_properties: SpaceDiscretization = None,
    *args,
    **kwargs,
):
    """Construct the interaction potential matrix for a two-body system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    N_points = space_properties.N_points
    dx = space_properties.dx
    x = np.linspace(space_properties.bounds[0], space_properties.bounds[1], N_points)

    interaction_matrix = np.zeros((N_points, N_points))
    for i in range(N_points):
        for j in range(N_points):
            interaction_matrix[i, j] = U(x[i], x[j], *args, **kwargs)
    return interaction_matrix


def interaction_matrix_many_body(
    U: callable,
    space_properties: SpaceDiscretization = None,
    N_particles=2,
    *args,
    **kwargs,
):
    """Construct the interaction potential matrix for a many-body system."""
    interaction_matrix = interaction_matrix_two_body(
        U, space_properties, *args, **kwargs
    )
    ordered_interaction_diag = interaction_matrix.flatten()
    # Generate the possible particle pairs without repetition

    idx_parti_pairs = tuple(itertools.combinations(range(N_parti), 2))


# YOU ARE HERE

a = SpaceDiscretization(N_points=2, L_box=2.0, PBC=True)

x = np.linspace(a.bounds[0], a.bounds[1], a.N_points)
print(x)


def V(x):
    return x


potential_matrix = potential_matrix_many_body(V, space_properties=a, N_particles=2)
print(potential_matrix)


psi = DiscreteState(N_states=2, N_particles=3, initial_state=[0, 1, 0])
print(psi)

N_parti = 4
parti_idx = np.arange(N_parti)
pairs = tuple(itertools.combinations(parti_idx, 2))
print(pairs)
