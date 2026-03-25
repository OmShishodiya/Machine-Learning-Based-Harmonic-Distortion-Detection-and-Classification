"""
kalman_filter.py

Implement basic 1D Kalman filtering for signal smoothing and distortion correction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleKalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_error):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.posteri_estimate = 0.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error = self.estimated_error + self.process_variance

        # Kalman Gain
        blending_factor = priori_error / (priori_error + self.measurement_variance)
        
        # Posteri state estimate
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        
        # Posteri error estimate
        self.estimated_error = (1 - blending_factor) * priori_error
        
        return self.posteri_estimate

def apply_kalman(signal):
    kf = SimpleKalmanFilter(0.001, 0.1, 1.0)
    kf.posteri_estimate = signal[0]
    return np.array([kf.update(s) for s in signal])

def main():
    try:
        df = pd.read_csv("all_signals_1000_1.csv", header=None)
        signals = df.iloc[:, 1:].values
    except Exception:
        t = np.linspace(0, 10, 800)
        signals = np.array([np.sin(t + np.random.rand()) for _ in range(500)])
        
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals)

    X_train, X_test = train_test_split(signals_scaled, test_size=0.2, random_state=42)
    X_test_unscaled = scaler.inverse_transform(X_test)
    X_test_pred_unscaled = np.zeros_like(X_test_unscaled)

    for i in range(len(X_test_unscaled)):
        # Apply KF directly on unscaled data
        X_test_pred_unscaled[i] = apply_kalman(X_test_unscaled[i])

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
    plt.subplot(3, 1, 1)
    plt.plot(original_signal, label='Distorted Input', color='black', alpha=0.5)
    plt.plot(reconstructed_signal, label='Reconstructed (Kalman Smoothed)', color='orange')
    plt.title(f'Sample {idx}: Kalman Filter Reconstruction')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(original_signal, label='Original (Distorted) Signal', color='black', alpha=0.5)
    plt.plot(distortion, label='Detected Distortion', color='red', alpha=0.7)
    plt.plot(corrected_signal, label='Corrected Signal (Orig + Anti-Wave)', color='green', linestyle='--')
    plt.title('Anti-Distortion Module Enforcement')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.hist((X_test_unscaled - X_test_pred_unscaled).flatten(), bins=50, color='purple')
    plt.title('Histogram of Reconstruction Errors')
    
    plt.tight_layout()
    plt.savefig("kalman_filter_results.png")
    print("Saved 'kalman_filter_results.png'")

if __name__ == '__main__':
    main()
