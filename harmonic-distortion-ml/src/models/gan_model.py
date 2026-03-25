"""
gan_signal_denoising.py

Generator-Discriminator based Adversarial architecture for signal denoising/correction.
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

class Generator(nn.Module):
    def __init__(self, seq_len):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, seq_len)
        )
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
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

    gen = Generator(timesteps).to(device)
    disc = Discriminator(timesteps).to(device)
    
    optimizer_G = optim.Adam(gen.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(disc.parameters(), lr=1e-4)
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    epochs = 100
    batch_size = 32
    train_losses_g = []

    gen.train()
    disc.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_g_loss = 0.0
        
        for i in range(0, X_train_tensor.size()[0], batch_size):
            clean_signals = X_train_tensor[permutation[i:i+batch_size]].to(device)
            # Add noise to simulate input distortion
            distorted_signals = clean_signals + 0.2 * torch.randn_like(clean_signals)
            
            valid = torch.ones(clean_signals.size(0), 1, device=device)
            fake = torch.zeros(clean_signals.size(0), 1, device=device)
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            gen_signals = gen(distorted_signals)
            
            # GAN Loss (fool the discriminator) + L1 loss (stay close to clean)
            g_loss_gan = criterion_GAN(disc(gen_signals), valid)
            g_loss_l1 = criterion_L1(gen_signals, clean_signals)
            g_loss = g_loss_gan + 10 * g_loss_l1
            g_loss.backward()
            optimizer_G.step()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_loss = criterion_GAN(disc(clean_signals), valid)
            fake_loss = criterion_GAN(disc(gen_signals.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            epoch_g_loss += g_loss.item() * clean_signals.size(0)
            
        epoch_g_loss /= X_train_tensor.size()[0]
        train_losses_g.append(epoch_g_loss)
        if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{epochs}], G Loss: {epoch_g_loss:.6f}")

    gen.eval()
    with torch.no_grad():
        noisy_test = X_test_tensor + 0.2 * torch.randn_like(X_test_tensor)
        X_test_pred = gen(noisy_test.to(device)).cpu()
        
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
    plt.plot(train_losses_g, color='blue', label='Generator Loss')
    plt.title('Training Loss Curve (GAN)')
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
    plt.savefig("gan_signal_denoising_results.png")
    print("Saved 'gan_signal_denoising_results.png'")

if __name__ == '__main__':
    main()
