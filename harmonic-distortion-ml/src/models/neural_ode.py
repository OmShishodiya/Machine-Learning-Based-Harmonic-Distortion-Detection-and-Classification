"""
neural_ode.py

A simple implementation of a continuous-depth Neural ODE for signal reconstruction.
Uses a basic Euler ODE solver for simplicity to avoid external dependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim

class ODEF(nn.Module):
    def __init__(self, dim):
        super(ODEF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )
    def forward(self, t, x):
        return self.net(x)

class EulerODESolver(nn.Module):
    def __init__(self, odef, steps=10):
        super(EulerODESolver, self).__init__()
        self.odef = odef
        self.steps = steps

    def forward(self, x_in, t_start=0.0, t_end=1.0):
        dt = (t_end - t_start) / self.steps
        x = x_in
        t = t_start
        for _ in range(self.steps):
            t_tensor = torch.tensor([t], dtype=torch.float32, device=x.device)
            x = x + self.odef(t_tensor, x) * dt
            t += dt
        return x

class NeuralODEAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(NeuralODEAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, 32)
        self.ode_block = EulerODESolver(ODEF(32), steps=5)
        self.decoder = nn.Linear(32, input_dim)

    def forward(self, x):
        h = self.encoder(x)
        h_t = self.ode_block(h)
        return self.decoder(h_t)

def main():
    device = torch.device('cpu')
    print("Using device:", device)

    try:
        df = pd.read_csv("all_signals_1000_1.csv", header=None)
        signals = df.iloc[:, 1:].values
    except Exception:
        t = np.linspace(0, 10, 800)
        signals = np.array([np.sin(t + np.random.rand()) for _ in range(500)])
        
    timesteps = signals.shape[1]
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals)

    X_train, X_test = train_test_split(signals_scaled, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model = NeuralODEAutoencoder(input_dim=timesteps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    batch_size = 32
    train_losses = []

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            batch_x = X_train_tensor[permutation[i:i+batch_size]].to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= X_train_tensor.size()[0]
        train_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad(): X_test_pred = model(X_test_tensor.to(device)).cpu()
        
    X_test_pred_unscaled = scaler.inverse_transform(X_test_pred.numpy())
    X_test_unscaled = scaler.inverse_transform(X_test)
    mse = mean_squared_error(X_test_unscaled, X_test_pred_unscaled)
    mae = mean_absolute_error(X_test_unscaled, X_test_pred_unscaled)
    print(f"\nMSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {np.sqrt(mse):.6f}")

    idx = 0
    original_signal = X_test_unscaled[idx]
    reconstructed_signal = X_test_pred_unscaled[idx]
    distortion = original_signal - reconstructed_signal
    anti_wave = -distortion
    corrected_signal = original_signal + anti_wave

    plt.figure(figsize=(15, 12))
    plt.subplot(4, 1, 1)
    plt.plot(train_losses, color='blue', label='Train Loss')
    plt.title('Training Loss Curve (Neural ODE)')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(original_signal, label='Distorted Input', color='black', alpha=0.5)
    plt.plot(reconstructed_signal, label='Reconstructed', color='orange')
    plt.title(f'Sample {idx}: Distorted vs Reconstructed')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(original_signal, label='Original (Distorted) Signal', color='black', alpha=0.5)
    plt.plot(distortion, label='Detected Distortion', color='red', alpha=0.7)
    plt.plot(corrected_signal, label='Corrected Signal (Orig + Anti-Wave)', color='green', linestyle='--')
    plt.title('Anti-Distortion Module Enforcement')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.hist((X_test_unscaled - X_test_pred_unscaled).flatten(), bins=50, color='purple')
    plt.title('Histogram of Reconstruction Errors')
    
    plt.tight_layout()
    plt.savefig("neural_ode_results.png")
    print("Saved 'neural_ode_results.png'")

if __name__ == '__main__':
    main()
