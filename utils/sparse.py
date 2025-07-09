import numpy as np
from numpy import ndarray
import itertools
from scipy.sparse import csr_matrix, kron, diags
from utils import chain_to_idx, idx_to_chain, SpaceDiscretization, load_constants

from constants import Constants


def load_constants():
    for name, value in Constants.__members__.items():
        globals()[name] = value.value


load_constants()


def kinetic_matrix_one_particle(
    space_properties: SpaceDiscretization = None,
) -> csr_matrix:
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
    kin_matrix = csr_matrix(kin_matrix)  # Convert to sparse matrix for efficiency
    return kin_matrix


def one_body_operator_many_particles(
    operator: ndarray, space_properties: SpaceDiscretization = None, N_particles=1
):
    """Construct the operator for a many-particle system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    N_points = space_properties.N_points
    II = csr_matrix(np.eye(N_points))  # Use sparse identity matrix for efficiency
    operator_one_p = operator
    operator_many_body = csr_matrix((N_points**N_particles, N_points**N_particles))

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
            term = kron(term, f, format="csr")

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

    pot_matrix = csr_matrix(pot_matrix)  # Convert to sparse matrix for efficiency
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

    N_points = space_properties.N_points
    interaction_matrix = interaction_matrix_two_body(
        U, space_properties, *args, **kwargs
    )
    # Generate the possible particle pairs without repetition

    idx_parti_pairs = list(itertools.combinations(range(N_particles), 2))

    interaction_diagonal = np.zeros((N_points**N_particles))
    for i in range(N_points**N_particles):
        # for each diagonal element of the hamiltonian:
        state = idx_to_chain(
            i, N_states=space_properties.N_points, N_particles=N_particles
        )
        # compute the state
        if len(state) != N_particles:
            raise ValueError(
                f"State {state} does not match the number of particles {N_particles}."
            )
        # Start computing the interaction term as the sum of the possible pairs
        U_total = 0.0
        # for each combination of particles
        for particle_pair in idx_parti_pairs:
            # Retrieve at what state are those particles
            x, y = state[particle_pair[0]], state[particle_pair[1]]
            U_total += interaction_matrix[x, y]
        interaction_diagonal[i] = U_total

    interaction_diagonal = diags(interaction_diagonal, offsets=0, format="csr")

    # interaction_diagonal = np.diag(interaction_diagonal)  # Convert to diagonal matrix
    # interaction_diagonal = csr_matrix(interaction_diagonal)  # Convert to sparse matrix for efficiency
    return interaction_diagonal


def total_hamiltonian(
    space_properties=None,
    V: callable = None,
    U: callable = None,
    N_particles=1,
    V_params: dict = None,
    U_params: dict = None,
):
    """Construct the total Hamiltonian for a many-particle system."""
    if not isinstance(space_properties, SpaceDiscretization):
        space_properties = SpaceDiscretization(N_points=100, L_box=2.0, PBC=True)

    kinetic_matrix = kinetic_matrix_many_particles(space_properties, N_particles)
    potential_matrix = potential_matrix_many_body(
        V, space_properties, N_particles, V_params if V_params is not None else {}
    )
    if N_particles > 1:
        interaction_matrix = interaction_matrix_many_body(
            U, space_properties, N_particles, U_params if U_params is not None else {}
        )
    else:
        interaction_matrix = np.zeros_like(potential_matrix)

    return kinetic_matrix + potential_matrix + interaction_matrix
