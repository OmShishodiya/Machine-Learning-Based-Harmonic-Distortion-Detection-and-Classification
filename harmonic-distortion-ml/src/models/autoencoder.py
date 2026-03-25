"""
autoencoder.py

Fully connected autoencoder for real-time sine wave distortion detection and correction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------------------------------------------------
# SIGNAL PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def compute_fft(signal, fs=1.0):
    """Compute standard FFT of a 1D signal."""
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
    magnitude = np.abs(fft_vals)
    return fft_freqs, magnitude

def compute_spectrogram(signal, fs=1.0):
    """Compute spectrogram using scipy.signal."""
    f, t, Sxx = scipy.signal.spectrogram(signal, fs=fs)
    return f, t, Sxx

def compute_energy(signal):
    """Calculate the total energy of a discrete 1D signal."""
    return np.sum(np.square(signal))

# -----------------------------------------------------------------------------
# NEURAL NETWORK ARCHITECTURE
# -----------------------------------------------------------------------------

class FullyConnectedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(FullyConnectedAutoencoder, self).__init__()
        # Encoder: compresses the signal 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32) # Bottleneck layer
        )
        
        # Decoder: reconstructs the signal
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    # Set execution explicitly for CPU
    device = torch.device('cpu')
    print("Using device:", device)

    # 1. Load dataset (Assumes rows are samples, first column is maybe an ID or parameter)
    file_path = "all_signals_1000_1.csv"
    try:
        df = pd.read_csv(file_path, header=None)
        print(f"Loaded dataset from {file_path}. Shape: {df.shape}")
        # Use columns 1 to end as signal, assuming col 0 is a label/parameter. 
        # If your data doesn't have labels in the first col, adjust this.
        signals = df.iloc[:, 1:].values
    except Exception as e:
        print(f"Error loading {file_path}. Details: {e}")
        # Fallback dummy sine wave dataset generator for standalone execution
        print("Generating a fallback synthetic sine wave dataset...")
        t = np.linspace(0, 10, 800)
        signals = np.array([np.sin(t + np.random.rand()) + 0.1 * np.random.randn(800) for _ in range(500)])
        
    num_samples = signals.shape[0]
    timesteps = signals.shape[1]

    # 2. Normalize Data
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals)

    # 3. Split into train/test (80/20)
    X_train, X_test = train_test_split(signals_scaled, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Convert iterables to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 4. Train the Model
    model = FullyConnectedAutoencoder(input_dim=timesteps).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    batch_size = 32
    train_losses = []

    print("Starting Training...")
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_train_tensor[indices].to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= X_train_tensor.size()[0]
        train_losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

    # 5. Perform predictions/reconstruction
    model.eval()
    with torch.no_grad():
        X_test_pred = model(X_test_tensor.to(device)).cpu()
        
    # Translate scaled predictions back to original distribution
    X_test_pred_unscaled = scaler.inverse_transform(X_test_pred.numpy())
    X_test_unscaled = scaler.inverse_transform(X_test)

    # 6. Compute Metrics
    # Using the unscaled data context
    mse = mean_squared_error(X_test_unscaled, X_test_pred_unscaled)
    mae = mean_absolute_error(X_test_unscaled, X_test_pred_unscaled)
    rmse = np.sqrt(mse)

    print("\n--- PERFORMANCE METRICS ---")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")

    # --- ANTI-DISTORTION MODULE ---
    # Formula given by user:
    # distortion = original - reconstructed
    # anti_wave = -distortion
    # corrected = original + anti_wave
    
    # We choose the first sample in the test set
    idx = 0
    original_signal = X_test_unscaled[idx]
    reconstructed_signal = X_test_pred_unscaled[idx]
    
    distortion = original_signal - reconstructed_signal
    anti_wave = -distortion
    corrected_signal = original_signal + anti_wave

    # Prepare specific errors (per step)
    errors = (X_test_unscaled - X_test_pred_unscaled).flatten()

    # 7. Visualize Results
    plt.figure(figsize=(15, 12))

    # Plot a) Training loss curve
    plt.subplot(4, 1, 1)
    plt.plot(train_losses, color='blue', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # Plot b) Original vs Reconstructed plot
    plt.subplot(4, 1, 2)
    plt.plot(original_signal, label='Original (Distorted) Signal', color='black', alpha=0.7)
    plt.plot(reconstructed_signal, label='Reconstructed (Clean) Signal', color='orange', linestyle='--')
    plt.title(f'Sample {idx}: Original vs Reconstructed')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot c) Reconstruction and Anti-Distortion
    plt.subplot(4, 1, 3)
    plt.plot(original_signal, label='Original (Distorted) Signal', color='black', alpha=0.5)
    plt.plot(distortion, label='Detected Distortion', color='red', alpha=0.7)
    plt.plot(corrected_signal, label='Corrected Signal (Orig + Anti-Wave)', color='green', linestyle='--')
    plt.title(f'Sample {idx}: Anti-Distortion Module Enforcement')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # Plot d) Histogram of Errors
    plt.subplot(4, 1, 4)
    plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Histogram of Reconstruction Errors (Test Set)')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("autoencoder_results.png")
    print("\nVisualizations saved to 'autoencoder_results.png'")
    plt.show()

if __name__ == '__main__':
    main()
