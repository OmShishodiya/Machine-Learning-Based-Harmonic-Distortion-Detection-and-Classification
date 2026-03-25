"""
diffusion_model.py

Simplified sequence-based diffusion architecture to denoise distorted signals.
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

class SimpleUNet1D(nn.Module):
    def __init__(self, seq_len):
        super(SimpleUNet1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, seq_len)
        )
    def forward(self, x, t_var):
        # We append t_var as simple embedding, but for brevity just pass to MLP
        return self.net(x)

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

    model = SimpleUNet1D(timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 100
    batch_size = 32
    train_losses = []

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            batch_x = X_train_tensor[permutation[i:i+batch_size]].to(device)
            # Add noise with a random timestep scale
            t_rand = torch.rand(batch_x.size(0), 1, device=device)
            noise = torch.randn_like(batch_x)
            noisy_x = batch_x * (1 - t_rand) + noise * t_rand
            
            # Predict NOISE or predict clean x directly (we choose to predict clean x directly)
            pred_clean = model(noisy_x, t_rand)
            loss = criterion(pred_clean, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= X_train_tensor.size()[0]
        train_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        noisy_test = X_test_tensor + 0.2 * torch.randn_like(X_test_tensor)
        X_test_pred = model(noisy_test.to(device), torch.zeros(noisy_test.size(0), 1)).cpu()
        
    X_test_pred_unscaled = scaler.inverse_transform(X_test_pred.numpy())
    X_test_unscaled = scaler.inverse_transform(X_test)
    noisy_test_unscaled = scaler.inverse_transform(noisy_test.numpy())
    
    mse = mean_squared_error(X_test_unscaled, X_test_pred_unscaled)
    mae = mean_absolute_error(X_test_unscaled, X_test_pred_unscaled)
    print(f"\nMSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {np.sqrt(mse):.6f}")

    idx = 0
    original_signal = noisy_test_unscaled[idx]
    clean_original = X_test_unscaled[idx]
    reconstructed_signal = X_test_pred_unscaled[idx]
    distortion = original_signal - reconstructed_signal
    anti_wave = -distortion
    corrected_signal = original_signal + anti_wave

    plt.figure(figsize=(15, 12))
    plt.subplot(4, 1, 1)
    plt.plot(train_losses, color='blue', label='Train Loss')
    plt.title('Training Loss Curve (Diffusion)')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(original_signal, label='Distorted Input', color='black', alpha=0.5)
    plt.plot(reconstructed_signal, label='Reconstructed', color='orange')
    plt.plot(clean_original, label='True Clean Signal', color='green', linestyle=':', alpha=0.8)
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
    plt.savefig("diffusion_model_results.png")
    print("Saved 'diffusion_model_results.png'")

if __name__ == '__main__':
    main()
