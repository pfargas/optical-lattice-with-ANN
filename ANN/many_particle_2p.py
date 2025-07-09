import qvarnet
from qvarnet.models.mlp import MLP
import torch
import itertools


Nh = 2  # Number of hidden neurons per layer
Nlayers = 3  # Number of hidden layers
layers = [2] + [Nh] * Nlayers + [1]  # Input layer, hidden layers, output layer

model = MLP(layer_dims=layers)


def testing_grads():
    x1 = torch.linspace(-1, 1, 100, requires_grad=True)
    x2 = torch.linspace(-1, 1, 100, requires_grad=True)
    # x = torch.stack([x1, x2], dim=1)  # Shape (100, 2)
    # generate the pairs

    y = model(x)
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    second_derivatives = []
    for i in range(N):
        # Compute first derivative w.r.t. x_i
        grad_psi = torch.autograd.grad(
            psi, x, grad_outputs=torch.ones_like(y), create_graph=True
        )[0]

        # Compute second derivative w.r.t. x_i
        d2psi_dxi2 = torch.autograd.grad(grad_psi[i], x[i], create_graph=True)[0]

        second_derivatives.append(d2psi_dxi2)

    print("Gradient shape:", dy_dx.shape)


testing_grads()


def train(
    x_train,
    optimizer,
    model,
    GRID_SPACING,
    epochs=1000,
    print_interval=100,
    callback=None,
    V0=1.0,
    K1=1.0,
):

    energy_history = []
    wf_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Run sampler

        psi = model(x_train)  # computes forward pass
        psi = psi / torch.sqrt(torch.trapezoid(torch.conj(psi) * psi, x_train, dim=0))
        T = kinetic_energy(psi, x_train)  # compute kinetic energy
        V_psi = V(x_train, **{"V0": V0, "K1": K1}) * psi
        H_psi = T + V_psi

        numerator = torch.trapezoid(torch.conj(psi) * H_psi, x_train, dim=0)
        denominator = torch.trapezoid(torch.conj(psi) * psi, x_train, dim=0)
        energy = numerator / denominator
        energy_copy = energy.clone().detach()
        loss = energy

        energy_history.append(energy.item())
        wf_history.append(psi.clone())

        penalty = 1e2
        # boundary conditions
        norm_psi = torch.trapezoid(psi.pow(2), x_train, dim=0)
        # loss    += 1e-1 * (norm_psi - 1)**2  # enforce normalization condition
        psi = psi / torch.sqrt(norm_psi + 1e-8)  # normalize wavefunction
        loss += penalty * (psi[0] - psi[-1]) ** 2  # penalize boundary conditions

        dpsi_0 = (psi[1] - psi[-1]) / (2 * GRID_SPACING)
        dpsi_end = (psi[0] - psi[-2]) / (2 * GRID_SPACING)
        loss += (
            penalty * (dpsi_0 - dpsi_end) ** 2
        )  # DERIVATIVE CONDITIONS IN THE PSI!!!!!

        loss.squeeze().backward()
        optimizer.step()

        if epoch % print_interval == 0:
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Energy: {energy_copy.item():.4f}, Num: {numerator.item():.4f}, Den: {denominator.item():.4f}"
            )

        if callback is not None:
            callback(epoch, loss, model)

            # If the callback indicates stopping, break the training loop
            if hasattr(callback, "stop_training") and callback.stop_training:
                model.load_state_dict(callback.best_model_state)
                print(f"Training stopped early at epoch {epoch+1}")
                break
