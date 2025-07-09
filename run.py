import argparse
import sys
import os
import datetime
import json
from scipy.sparse.linalg import eigsh
from utils import *

parser = argparse.ArgumentParser(
    description="Sparse many-body Hamiltonian diagonalization."
)
parser.add_argument("--N_particles", type=int, default=2, help="Number of particles.")
parser.add_argument(
    "--N_points", type=int, default=100, help="Number of discretized points."
)
parser.add_argument(
    "--L_box", type=float, default=2.0, help="Length of the box for discretization."
)
parser.add_argument(
    "--PBC",
    action="store_true",
    help="Use periodic boundary conditions.",
    default=True,
)
parser.add_argument(
    "--n_eigvals",
    type=int,
    default=5,
    help="Number of eigenvalues to compute.",
)

args = parser.parse_args()

for arg in vars(args).items():
    print(f"{arg[0]}: {arg[1]}")

N_particles = args.N_particles
N_points = args.N_points
L_box = args.L_box
PBC = args.PBC
n_eigvals = args.n_eigvals

a = SpaceDiscretization(N_points=N_points, L_box=L_box, PBC=PBC)


def V(x, kwargs):
    """Potential function V(x) for the system."""
    V0 = kwargs["V0"] if "V0" in kwargs else 1.0
    K1 = kwargs["K1"] if "K1" in kwargs else 1.0
    return V0 * np.sin(K1 * x) ** 2


def U(x1, x2, kwargs):
    return np.exp(-((x1 - x2) ** 2))


H = total_hamiltonian(space_properties=a, V=V, U=U, N_particles=N_particles)
print("Shape of Hamiltonian:", H.shape)

eigvals, eigvecs = eigsh(H, k=n_eigvals, which="SM")
print("Eigenvalues:", eigvals)

# save in output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

directory_name = f"sparse_many_body_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

final_directory = os.path.join(output_dir, directory_name)

if not os.path.exists(final_directory):
    os.makedirs(final_directory)

# Save parameters in a JSON file
params = {
    "N_particles": N_particles,
    "N_points": N_points,
    "L_box": L_box,
    "PBC": PBC,
}
with open(os.path.join(final_directory, "params.json"), "w") as f:
    json.dump(params, f, indent=4)

np.save(os.path.join(final_directory, "eigvals.npy"), eigvals)
np.save(os.path.join(final_directory, "eigvecs.npy"), eigvecs)
