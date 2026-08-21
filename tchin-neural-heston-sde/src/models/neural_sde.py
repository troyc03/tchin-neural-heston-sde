import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def generate_heston_data(
    NoOfPaths=5000,
    NoOfSteps=100,
    T=1.0,
    r=0.05,
    S_0=100.0,
    kappa=1.5,
    gamma=0.3,
    rho=-0.6,
    vbar=0.04,
    v0=0.04,
    seed=42
):
    np.random.seed(seed)

    dt = T / NoOfSteps

    S = np.zeros((NoOfPaths, NoOfSteps + 1), dtype=np.float32)
    V = np.zeros((NoOfPaths, NoOfSteps + 1), dtype=np.float32)

    S[:, 0] = S_0
    V[:, 0] = v0

    for i in range(NoOfSteps):
        Z1 = np.random.normal(size=NoOfPaths)
        Z2 = np.random.normal(size=NoOfPaths)

        ZW = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

        V_pos = np.maximum(V[:, i], 0.0)

        V[:, i + 1] = (
            V[:, i]
            + kappa * (vbar - V_pos) * dt
            + gamma * np.sqrt(V_pos * dt) * Z1
        )

        V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)

        S[:, i + 1] = (
            S[:, i]
            + r * S[:, i] * dt
            + np.sqrt(V_pos) * S[:, i] * np.sqrt(dt) * ZW
        )

        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-8)

    return (
        torch.tensor(S, dtype=torch.float32),
        torch.tensor(V, dtype=torch.float32),
        dt
    )

# Generate data
print("Generating Heston paths...")

S_paths, V_paths, dt = generate_heston_data()

r = 0.05

S_curr = S_paths[:, :-1].reshape(-1, 1)
V_curr = V_paths[:, :-1].reshape(-1, 1)

# True conditional drift of Heston stock process
target_drift = r * S_curr

# Neural SDE drift model
class NeuralSDEDrift(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, S, V):
        S_scaled = S / 100.0
        V_scaled = V / 0.04

        x = torch.cat([S_scaled, V_scaled], dim=1)

        return self.net(x) * S


# Training
model = NeuralSDEDrift()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

epochs = 300
loss_history = []

print("Training the Neural SDE drift...")

for epoch in range(1, epochs + 1):

    optimizer.zero_grad()

    predicted_drift = model(
        S_curr,
        V_curr
    )

    loss = loss_fn(
        predicted_drift,
        target_drift
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    loss_history.append(loss.item())

    if epoch == 1 or epoch % 25 == 0:
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"MSE Loss: {loss.item():.6f}"
        )



# Loss plot
plt.figure(figsize=(9, 4.5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    color="#1f77b4",
    linewidth=2,
    label="MSE Loss"
)

plt.yscale("log")
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.title("Neural SDE Drift Training")

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.5
)

plt.legend()
plt.tight_layout()
plt.show()


# Compare learned drift against true drift
model.eval()

with torch.no_grad():

    V_test = torch.full_like(
        S_curr,
        0.04
    )

    predicted = model(
        S_curr,
        V_test
    )

    true_drift = r * S_curr


sort_idx = torch.argsort(S_curr[:, 0])

S_sorted = S_curr[sort_idx]
predicted_sorted = predicted[sort_idx]
true_sorted = true_drift[sort_idx]


plt.figure(figsize=(9, 5))

plt.scatter(
    S_sorted.numpy(),
    predicted_sorted.numpy(),
    s=3,
    alpha=0.2,
    label="Neural Network"
)

plt.plot(
    S_sorted.numpy(),
    true_sorted.numpy(),
    color="red",
    linewidth=2,
    label=r"True Drift $rS$"
)

plt.xlabel(r"$S_t$")
plt.ylabel("Drift")
plt.title("Learned Heston Stock Drift")

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()
plt.tight_layout()
plt.show()


# Sample predictions
sample_S = torch.tensor(
    [[80.0], [90.0], [100.0], [110.0], [120.0]]
)

sample_V = torch.full_like(
    sample_S,
    0.04
)

with torch.no_grad():
    sample_prediction = model(
        sample_S,
        sample_V
    )

print("\nSample drift predictions:")

for s, prediction in zip(
    sample_S.flatten(),
    sample_prediction.flatten()
):
    true_value = r * s.item()

    print(
        f"S = {s.item():6.1f} | "
        f"True = {true_value:7.3f} | "
        f"NN = {prediction.item():7.3f}"
    )