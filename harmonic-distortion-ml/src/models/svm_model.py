"""
one_class_svm.py

Uses One-Class SVM to model the normal distribution of signal points, 
correcting out-of-distribution values similarly to IF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d

def main():
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
    
    # Train OCSVM on a subset of un-anomalous points or directly on values
    X_train_flat = X_train.flatten().reshape(-1, 1)
    
    # Random sub-sample to speed up SVM training if it's too large
    if len(X_train_flat) > 20000:
        np.random.seed(42)
        idx = np.random.choice(len(X_train_flat), 20000, replace=False)
        X_train_subset = X_train_flat[idx]
    else:
        X_train_subset = X_train_flat

    print("Training One-Class SVM...")
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    svm.fit(X_train_subset)

    X_test_unscaled = scaler.inverse_transform(X_test)
    X_test_pred_unscaled = np.zeros_like(X_test_unscaled)

    for i in range(len(X_test)):
        sig = X_test[i].reshape(-1, 1)
        preds = svm.predict(sig)
        
        normal_idx = np.where(preds == 1)[0]
        anomaly_idx = np.where(preds == -1)[0]
        
        sig_clean = sig.copy().flatten()
        if len(normal_idx) > 1 and len(anomaly_idx) > 0:
            f = interp1d(normal_idx, sig_clean[normal_idx], bounds_error=False, fill_value="extrapolate")
            sig_clean[anomaly_idx] = f(anomaly_idx)
            
        X_test_pred_unscaled[i] = scaler.inverse_transform(sig_clean.reshape(1, -1))[0]

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
    plt.plot(reconstructed_signal, label='Reconstructed (OCSVM Cleaned)', color='orange')
    plt.title(f'Sample {idx}: OCSVM Distorted vs Reconstructed')
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
    plt.savefig("one_class_svm_results.png")
    print("Saved 'one_class_svm_results.png'")

if __name__ == '__main__':
    main()
