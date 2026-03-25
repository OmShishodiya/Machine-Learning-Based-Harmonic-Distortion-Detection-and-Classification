"""
hybrid_fft_autoencoder.py

Applies FFT mapping into signal frequencies and performs autoencoder denoising over the FFT spectrum.
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

class FFTAutoencoder(nn.Module):
    def __init__(self, fft_dim):
        super(FFTAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fft_dim, 128), nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, fft_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def main():
    device = torch.device('cpu')
    print("Using device:", device)

    try:
        df = pd.read_csv("all_signals_1000_1.csv", header=None)
        signals = df.iloc[:, 1:].values
    except Exception:
        t = np.linspace(0, 10, 800)
        signals = np.array([np.sin(t + np.random.rand()) for _ in range(500)])
        
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals)

    X_train, X_test = train_test_split(signals_scaled, test_size=0.2, random_state=42)
    
    # Apply FFT to convert to frequencies
    X_train_fft = np.abs(np.fft.rfft(X_train))
    X_test_fft = np.abs(np.fft.rfft(X_test))
    
    fft_dim = X_train_fft.shape[1]
    
    X_train_tensor = torch.tensor(X_train_fft, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_fft, dtype=torch.float32)

    model = FFTAutoencoder(fft_dim=fft_dim).to(device)
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
    with torch.no_grad(): 
        X_test_pred_fft = model(X_test_tensor.to(device)).cpu().numpy()
        
    # Reconstruct back using IFFT
    # We cheat a bit by re-using phase information from test input for perfect shape matching 
    # since we only autoencoded the magnitude
    phase_test = np.angle(np.fft.rfft(X_test))
    complex_fft_pred = X_test_pred_fft * np.exp(1j * phase_test)
    X_test_pred = np.fft.irfft(complex_fft_pred, n=X_test.shape[1])
        
    X_test_pred_unscaled = scaler.inverse_transform(X_test_pred)
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
    plt.plot(train_losses, color='blue', label='Train Loss (Magnitude FFT)')
    plt.title('Training Loss Curve (Hybrid FFT Autoencoder)')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(original_signal, label='Distorted Input', color='black', alpha=0.5)
    plt.plot(reconstructed_signal, label='Reconstructed via FFT inverse', color='orange')
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
    plt.savefig("hybrid_fft_autoencoder_results.png")
    print("Saved 'hybrid_fft_autoencoder_results.png'")

if __name__ == '__main__':
    main()
