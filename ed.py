# Exact diagonalization utils

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


class ExactDiagonalizationParams:
    def __init__(self,N_particles: int, N_sites: int, L_BOX: float, PBC:bool = True):
        self.N_particles = N_particles
        self.N_sites = N_sites
        self.L_BOX = L_BOX
        self.PBC = PBC
        self.GRID_SPACING = L_BOX / N_sites

@dataclass
class Constants:
    HBAR: float = 1.0  # Reduced Planck's constant
    MASS: float = 1.0


def II(N):
    """Identity operator for N sites."""
    return np.eye(N)

def K_mat(N, params: ExactDiagonalizationParams, constants: Constants = Constants(), debug = False):
    """Kinetic energy operator."""
    kin_constant = -0.5 * constants.HBAR**2 / (constants.MASS * params.GRID_SPACING**2)
    if debug:
        kin_constant = 1
    K = np.zeros((N, N))
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    K += np.diag(main_diag)
    K += np.diag(off_diag, k=1)
    K += np.diag(off_diag, k=-1)
    if params.PBC:
        K[0, -1] = 1
        K[-1, 0] = 1
    return kin_constant * K

def one_body_full_hamiltonian(params: ExactDiagonalizationParams, constants: Constants = Constants(), debug = False):
    """One-body Hamiltonian."""
    N = params.N_sites
    P = params.N_particles
    K = K_mat(N, params, constants, debug=debug)
    # K = 2 * np.diag(N, k=1)  # Placeholder for kinetic energy matrix
    H = np.zeros((N**P, N**P))
    # construct the sum of kinetic energy operators for each particle
    # for i in range(P):
        # Create a block diagonal matrix for each particle's kinetic energy
        # block = np.zeros((N**P, N**P))
        # for j in range(N**P):
            # block[j, j] = K[j % N, j % N]
        # H += block
    # construct the sum of kinetic energy operators for each particle with kron products
    for i in range(P):
        # sum loop
        aux_mat = np.eye(1)
        for j in range(P):
            if j<i:
                aux_mat = np.kron(aux_mat, II(N))
            elif i==j:
                aux_mat = np.kron(aux_mat, K)
            else:
                aux_mat = np.kron(aux_mat, II(N))
    
        H += aux_mat

    return H


# test H

def test_kin_hamiltonian():
    params = ExactDiagonalizationParams(N_particles=2, N_sites=2, L_BOX=10.0, PBC=True) 
    H = one_body_full_hamiltonian(params, debug=True)
    true_val = np.array([[-4.,  1.,  1.,  0.],[ 1., -4.,  0.,  1.],[ 1.,  0., -4.,  1.],[ 0.,  1.,  1., -4.]])
    assert np.allclose(H, true_val), "Hamiltonian does not match expected values."


# parameters = ExactDiagonalizationParams(N_particles=2, N_sites=2, L_BOX=10.0, PBC=True)

# H = one_body_full_hamiltonian(parameters)
# print("Hamiltonian matrix shape:", H.shape)
# assert H.shape == (4, 4), "Hamiltonian matrix should be 4x4 for 2 particles and 2 sites."

# print("Hamiltonian matrix:\n", H)

test_kin_hamiltonian()