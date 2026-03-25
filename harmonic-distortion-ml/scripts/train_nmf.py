"""
Non-Negative Matrix Factorization (NMF) on Multivariate Signal Data (Unsupervised)
Decomposes data into non-negative basis components and coefficients.
Uses only: pandas, numpy, scikit-learn, matplotlib.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load the CSV dataset
# ---------------------------------------------------------------------------
# Unsupervised: no target column. Each row = one signal instance,
# each column = amplitude at a fixed frequency.
df = pd.read_csv("all_signals_1000_1.csv")
X = df.values  # Shape: (n_samples, n_features)

# ---------------------------------------------------------------------------
# 2. Ensure all values are non-negative (clip small negatives from noise)
# ---------------------------------------------------------------------------
# NMF requires non-negative input. Clip any negative values to zero.
X = np.clip(X, 0, None)

# ---------------------------------------------------------------------------
# 3. Scale the data using MinMaxScaler (required for NMF)
# ---------------------------------------------------------------------------
# MinMaxScaler maps values to [0, 1], preserving non-negativity. StandardScaler
# can produce negatives, so MinMaxScaler is the right choice for NMF.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
n_samples, n_features = X_scaled.shape

# ---------------------------------------------------------------------------
# 4. Apply NMF
# ---------------------------------------------------------------------------
# Why NMF is suitable: Signal amplitudes are non-negative. NMF finds additive
# parts (basis components) that sum to approximate the data, leading to
# interpretable "building blocks" (e.g. spectral patterns) and parts-based
# representation. Difference from PCA: PCA allows negative loadings and is
# variance-oriented; NMF is non-negative and additive, often more interpretable
# for magnitude data (e.g. spectra, intensities).
n_components = 8
nmf = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=500)
W = nmf.fit_transform(X_scaled)   # Coefficients: (n_samples, n_components)
H = nmf.components_               # Basis: (n_components, n_features)

# ---------------------------------------------------------------------------
# 5. Reconstruct the original data from NMF: X ≈ W @ H
# ---------------------------------------------------------------------------
X_reconstructed = W @ H

# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
mse_reconstruction = mean_squared_error(X_scaled, X_reconstructed)
rmse_reconstruction = np.sqrt(mse_reconstruction)
var_original = np.var(X_scaled)
accuracy = 100 * (1 - mse_reconstruction / var_original) if var_original != 0 else 0.0

print("=" * 50)
print("NMF RECONSTRUCTION METRICS")
print("=" * 50)
print(f"Mean Squared Error (MSE):  {mse_reconstruction:.6f}")
print(f"Root Mean Squared Error (RMSE):  {rmse_reconstruction:.6f}")
print(f"Reconstruction Accuracy (%):  {accuracy:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

# (a) Original vs reconstructed signal for one sample
ax1 = fig.add_subplot(2, 2, 1)
sample_idx = 0
feat_idx = np.arange(n_features)
ax1.plot(feat_idx, X_scaled[sample_idx], "b-", alpha=0.8, label="Original")
ax1.plot(feat_idx, X_reconstructed[sample_idx], "r--", alpha=0.8, label="Reconstructed")
ax1.set_xlabel("Feature index")
ax1.set_ylabel("Amplitude (scaled)")
ax1.set_title("(a) Original vs Reconstructed (one sample)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Reconstruction error distribution (per-sample MSE, histogram)
ax2 = fig.add_subplot(2, 2, 2)
mse_per_sample = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
ax2.hist(mse_per_sample, bins=30, edgecolor="k", alpha=0.7)
ax2.set_xlabel("Reconstruction MSE (per sample)")
ax2.set_ylabel("Frequency")
ax2.set_title("(b) Reconstruction Error Distribution")
ax2.grid(True, alpha=0.3, axis="y")

# (c) Learned NMF basis components (each row of H is one component)
# Interpretation: Each basis is a "building block" pattern across frequencies;
# signals are positive combinations of these patterns. Peaks show which
# frequencies contribute to that component.
ax3 = fig.add_subplot(2, 2, 3)
for i in range(n_components):
    ax3.plot(feat_idx, H[i], alpha=0.8, label=f"Component {i+1}")
ax3.set_xlabel("Feature index")
ax3.set_ylabel("Component amplitude")
ax3.set_title("(c) NMF Basis Components")
ax3.legend(loc="upper right", fontsize=7)
ax3.grid(True, alpha=0.3)

# (d) Per-feature reconstruction error (bar plot)
ax4 = fig.add_subplot(2, 2, 4)
mse_per_feature = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
n_bars = min(30, n_features)
ax4.bar(range(n_bars), mse_per_feature[:n_bars], align="center")
ax4.set_xlabel("Feature index")
ax4.set_ylabel("MSE")
ax4.set_title("(d) Per-Feature Reconstruction Error")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("nmf_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'nmf_evaluation.png'")
