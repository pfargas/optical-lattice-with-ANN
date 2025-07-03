import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    import qvarnet
    from qvarnet.models.mlp import MLP
except ImportError:
    print("If variational approach, please install qvarnet.")

from tqdm import tqdm
import copy

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

print(f"Using device: {device}")

# PARAMETERS
HBAR = 1.0
MASS = 1.0
V0   = 1.0
K1   = 1.0
PBC  = True

N_POINTS = 1_000    # total number of points in (-L/2,+L/2)
N_PERIODS_FROM_CENTRE_TO_RIGHT = 10 # number of periods from the center to the right edge
L_BOX = 2*torch.pi/K1 * N_PERIODS_FROM_CENTRE_TO_RIGHT
GRID_SPACING = L_BOX / N_POINTS

class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_energy = None
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.stop_training = False

    def __call__(self, epoch, energy, model):
        # Initialize the best energy if it's the first epoch
        if self.best_energy is None:
            self.best_energy = energy
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.epochs_without_improvement = 0
        else:
            # Check if the energy has improved by more than min_delta
            if energy < (self.best_energy - self.min_delta):
                self.best_energy = energy
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        # If there is no improvement for the specified patience, stop training
        if self.epochs_without_improvement >= self.patience:
            print(f"Stopping training after {epoch+1} epochs due to no improvement in energy.")
            self.stop_training = True

def V(x,V0,K1):
    """Potential function."""
    return V0 * torch.sin(K1 * x).pow(2)


def hamiltonian_matrix(N_POINTS, GRID_SPACING, PBC, potential_parameters, x):
    kin_coeff = -HBAR**2 / (2 * MASS * GRID_SPACING**2)
    kin_main_diag = -2*torch.ones(N_POINTS, device=device)  # Main diagonal for kinetic energy
    kin_off_diags = 1*torch.ones(N_POINTS-1, device=device)  # Off-diagonal elements for kinetic energy
    # kin_matrix = torch.zeros((N_POINTS, N_POINTS))
    kin_matrix = torch.diag(kin_main_diag).to(device)
    kin_matrix += torch.diag(kin_off_diags, diagonal=1)
    kin_matrix += torch.diag(kin_off_diags, diagonal=-1)
    if PBC:
        kin_matrix[0, -1] = 1
        kin_matrix[-1, 0] = 1
    pot_matrix = torch.zeros((N_POINTS, N_POINTS), device=device)
    for i in range(N_POINTS):
        pot_matrix[i, i] = V(x[i],**potential_parameters)
        # TO TRY HARMONIC POTENTIAL COMMENT OUT ABOVE LINE AND UNCOMMENT BELOW
        # pot_matrix[i, i] = 0.5*MASS* x[i]**2  # Harmonic potential
    return kin_coeff * kin_matrix + pot_matrix


def kinetic_energy(psi, x):
    # First derivative
    dpsi_dx = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                  create_graph=True, retain_graph=True)[0]
    # Second derivative
    d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx),
                                    create_graph=True, retain_graph=True)[0]
    return -0.5 * d2psi_dx2


x_train = torch.linspace(-L_BOX/2, L_BOX/2, N_POINTS).view(-1, 1).to(device)
x_train.requires_grad = True

# ED_____________________________________
ed_eigvals, ed_eigstates = torch.linalg.eigh(hamiltonian_matrix(N_POINTS, GRID_SPACING, PBC, {'V0': V0, 'K1': K1}, x_train))
# VARIATIONAL ___________________________

epochs  = 30_000
Nh      = 100 #100
NLayers = 2
lr      = 0.001
activation_function = "tanh"

energy_history = []
wf_history = []

layer_dims = [1] + [Nh] * NLayers + [1]

model = MLP(layer_dims=layer_dims, activation=activation_function)
model.to(device)

# init weights
for param in model.parameters():
    if len(param.shape) > 1:  # only initialize weights, not biases
        torch.nn.init.xavier_uniform_(param)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

callback = EarlyStoppingCallback(patience=5000, min_delta=1e-4)

print_interval = 100

for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    # Run sampler

    psi = model(x_train)     # computes forward pass
    psi = psi / torch.sqrt(torch.trapezoid(torch.conj(psi) * psi, x_train, dim=0))
    T = kinetic_energy(psi,x_train)  # compute kinetic energy
    V_psi = V(x_train,**{'V0': V0, 'K1': K1}) * psi
    H_psi = T + V_psi

    numerator   = torch.trapezoid(torch.conj(psi) * H_psi, x_train, dim=0)
    denominator = torch.trapezoid(torch.conj(psi) * psi, x_train, dim=0)
    energy      = numerator / denominator
    energy_copy = energy.clone().detach()
    loss        = energy

    energy_history.append(energy.item())
    wf_history.append(psi.clone())

    penalty = 1e2
    # boundary conditions
    norm_psi = torch.trapezoid(psi.pow(2), x_train, dim=0)
    #loss    += 1e-1 * (norm_psi - 1)**2  # enforce normalization condition
    psi      = psi / torch.sqrt(norm_psi + 1e-8)  # normalize wavefunction
    loss    += penalty * (psi[0] - psi[-1])**2 # penalize boundary conditions

    dpsi_0   = (psi[1]-psi[-1])/(2*GRID_SPACING)
    dpsi_end = (psi[0]-psi[-2])/(2*GRID_SPACING)
    loss    += penalty*(dpsi_0-dpsi_end)**2 # DERIVATIVE CONDITIONS IN THE PSI!!!!!
    
    loss.squeeze().backward()
    optimizer.step()

    # if epoch % print_interval == 0:
    #     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Energy: {energy_copy.item():.4f}, Num: {numerator.item():.4f}, Den: {denominator.item():.4f}")

    if callback is not None:
        callback(epoch, loss, model)

        # If the callback indicates stopping, break the training loop
        if hasattr(callback, 'stop_training') and callback.stop_training:
            model.load_state_dict(callback.best_model_state)
            print(f"Training stopped early at epoch {epoch+1}")
            break
    
# print(energy)
# model.load_state_dict(callback.best_model_state)

ed_ground_state = ed_eigstates[:, 0].view(-1, 1)

plt.plot(x_train.cpu().detach().numpy(), psi.detach().cpu().numpy()**2, label='MLP Wavefunction', alpha=0.5)
norm = torch.trapezoid(psi.pow(2), x_train, dim=0)
print(f'Norm of wavefunction: {norm.item()}')
print(f"shape of ed_eigstates: {ed_eigstates.shape}")
norm_other = torch.trapezoid(ed_ground_state.pow(2), x_train, dim=0)
print(f"shape of norm_other: {norm_other.shape}")
print(f"type of norm_other: {type(norm_other)}")
print(f'Norm of eigenstate: {norm_other.item()}')
psi_ed = ed_ground_state.clone().detach()/ torch.sqrt(norm_other + 1e-8)  # normalize eigenstate
new_norm = torch.trapezoid(psi_ed.pow(2), x_train, dim=0)
print(f'New norm of eigenstate: {new_norm.item()}')
plt.plot(x_train.detach().cpu().numpy(), psi_ed.cpu().detach().numpy()**2, label='ED Wavefunction', alpha=0.5)
plt.plot(x_train.cpu().detach().numpy(), V(x_train,V0,K1).cpu().detach().numpy(), label='Potential V(x)', color='gray', linestyle='dashed', alpha=1, linewidth=0.5)

print("......................")
print(f'Energy from ED: {ed_eigvals[0].item()}')
print(f'Energy from VMC: {energy_copy.item()}')
plt.xlabel('x')
plt.ylabel('Wavefunction')
plt.legend()
plt.show();