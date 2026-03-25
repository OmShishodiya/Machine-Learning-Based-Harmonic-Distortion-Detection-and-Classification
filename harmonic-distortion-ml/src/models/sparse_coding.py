"""
sparse_coding.py

Dictionary learning for sparse coding to extract clean sine components.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import DictionaryLearning

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

    # Dictionary learning is expensive, so limit components and iterations
    n_components = 15
    print(f"Training Dictionary Learning ({n_components} components)...")
    dl = DictionaryLearning(n_components=n_components, alpha=1.0, max_iter=100, random_state=42)
    dl.fit(X_train)
    
    # Reconstruct test signals
    X_test_transform = dl.transform(X_test)
    X_test_pred = np.dot(X_test_transform, dl.components_)

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
    plt.subplot(3, 1, 1)
    plt.plot(original_signal, label='Distorted Input', color='black', alpha=0.5)
    plt.plot(reconstructed_signal, label='Reconstructed (Sparse Coding)', color='orange')
    plt.title(f'Sample {idx}: Sparse Coding Reconstruction')
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
    plt.savefig("sparse_coding_results.png")
    print("Saved 'sparse_coding_results.png'")

if __name__ == '__main__':
    main()
