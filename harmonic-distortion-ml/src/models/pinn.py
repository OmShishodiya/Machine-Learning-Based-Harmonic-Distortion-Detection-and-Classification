import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Data
# The dataset has 2000 rows. Col 0 is parameter 'p', Col 1-800 is 'u(t)'
# We will use a subset to train quickly and show visualization.
df = pd.read_csv('all_signals_1000_1.csv', header=None)
print("Data loaded. Shape:", df.shape)

p_values = df.iloc[:, 0].values
u_values = df.iloc[:, 1:].values
num_samples = df.shape[0]
num_time_steps = u_values.shape[1]

# Normalize time and data for better training
t_arr = np.linspace(0, 1, num_time_steps)
p_norm = (p_values - p_values.min()) / (p_values.max() - p_values.min() + 1e-8)
u_norm = (u_values - u_values.min()) / (u_values.max() - u_values.min() + 1e-8)

# Create train dataset (t, p) -> u
T, P = np.meshgrid(t_arr, p_norm)
T_flat = T.flatten()[:, None]
P_flat = P.flatten()[:, None]
U_flat = u_norm.flatten()[:, None]

# Convert to PyTorch tensors
t_tensor = torch.tensor(T_flat, dtype=torch.float32, requires_grad=True).to(device)
p_tensor = torch.tensor(P_flat, dtype=torch.float32, requires_grad=True).to(device)
u_tensor = torch.tensor(U_flat, dtype=torch.float32).to(device)

# Downsample for faster training
indices = np.random.choice(len(T_flat), size=int(0.02 * len(T_flat)), replace=False)
t_train = t_tensor[indices].clone().detach().requires_grad_(True)
p_train = p_tensor[indices].clone().detach().requires_grad_(True)
u_train = u_tensor[indices].clone().detach()

# 2. Define PINN Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Learnable physical parameters for an assumed ODE: u_tt + theta1 * u_t + theta2 * u = 0
        self.theta1 = nn.Parameter(torch.tensor([0.1], device=device))
        self.theta2 = nn.Parameter(torch.tensor([1.0], device=device))

    def forward(self, t, p):
        x = torch.cat([t, p], dim=1)
        return self.net(x)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 3. Training Loop
epochs = 300
loss_data_history = []
loss_phys_history = []
theta1_history = []
theta2_history = []

print("Starting training...")
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Data loss
    u_pred = model(t_train, p_train)
    mse_data = torch.mean((u_pred - u_train)**2)
    
    # Physics loss
    # Compute derivatives u_t and u_tt
    u_t = torch.autograd.grad(
        u_pred, t_train,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True, retain_graph=True
    )[0]
    
    u_tt = torch.autograd.grad(
        u_t, t_train,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True, retain_graph=True
    )[0]
    
    # ODE residual: u_tt + theta1*u_t + theta2*u = 0
    f_pred = u_tt + model.theta1 * u_t + model.theta2 * u_pred
    mse_phys = torch.mean(f_pred**2)
    
    # Total loss
    loss = mse_data + 1e-4 * mse_phys
    
    loss.backward()
    optimizer.step()
    
    loss_data_history.append(mse_data.item())
    loss_phys_history.append(mse_phys.item())
    theta1_history.append(model.theta1.item())
    theta2_history.append(model.theta2.item())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}: Total Loss={loss.item():.5f}, Data Loss={mse_data.item():.5f}, Phys Loss={mse_phys.item():.5f}, Theta1={model.theta1.item():.3f}, Theta2={model.theta2.item():.3f}")

print("Training finished!")
print(f"Final Data MSE: {mse_data.item():.5f}")
# Accuracy representation based on relative L2 error
rel_error = torch.linalg.norm(u_pred - u_train) / torch.linalg.norm(u_train)
accuracy = max(0.0, 100.0 * (1.0 - rel_error.item()))
print(f"Approximated Accuracy (1 - Rel Error): {accuracy:.2f}%")

# 4. Visualizations
os.makedirs('visualizations', exist_ok=True)

# Plot 1: Loss curves
plt.figure()
plt.plot(loss_data_history, label='Data Loss')
plt.plot(loss_phys_history, label='Physics Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.title('PINN Training Loss')
plt.legend()
plt.savefig('visualizations/loss_curves.png')
plt.close()

# Plot 2: Parameter convergence
plt.figure()
plt.plot(theta1_history, label='Theta 1 (damping)')
plt.plot(theta2_history, label='Theta 2 (stiffness)')
plt.xlabel('Epochs')
plt.ylabel('Parameter Value')
plt.title('Discovered Physical Parameters')
plt.legend()
plt.savefig('visualizations/parameter_convergence.png')
plt.close()

# Plot 3: Signal Prediction Comparison
# Predict a specific sequence (Row 0)
idx = 0
test_t = torch.tensor(t_arr[:, None], dtype=torch.float32).to(device)
test_p = torch.tensor(np.full_like(t_arr, p_norm[idx])[:, None], dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    pred_u = model(test_t, test_p).cpu().numpy().flatten()

true_u = u_norm[idx]

plt.figure(figsize=(10, 5))
plt.plot(t_arr, true_u, label='True Signal', color='blue')
plt.plot(t_arr, pred_u, label='PINN Prediction', color='red', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Normalized Amplitude')
plt.title(f'Signal Comparison for parameter p={p_values[idx]}')
plt.legend()
plt.savefig('visualizations/signal_prediction.png')
plt.close()

print("Visualizations saved in 'visualizations/' folder.")
