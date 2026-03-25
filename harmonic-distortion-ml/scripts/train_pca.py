"""
PCA Analysis of Multivariate Frequency-Domain Signal Data
Unsupervised dimensionality reduction and reconstruction.
Uses only: pandas, numpy, scikit-learn, matplotlib.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load the CSV dataset
# ---------------------------------------------------------------------------
# Each row = one signal instance, each column = amplitude at a fixed frequency.
# No target column: this is unsupervised (we don't predict a single output).
# Why regression was invalid: There is no single target variable to predict;
# we have many frequency amplitudes per sample. Regression assumes one output
# per input; here we want to capture structure across all frequencies jointly.
df = pd.read_csv("all_signals_1000_1.csv")

# ---------------------------------------------------------------------------
# 2. Extract the full data matrix (all columns are features)
# ---------------------------------------------------------------------------
X = df.values  # Shape: (n_samples, n_frequencies)

# ---------------------------------------------------------------------------
# 3. Standardize the data using StandardScaler
# ---------------------------------------------------------------------------
# PCA is sensitive to scale; frequencies with larger variance would dominate
# otherwise. Standardization puts all dimensions on equal footing.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 4. Apply PCA and analyze explained variance
# ---------------------------------------------------------------------------
# Why PCA is appropriate: We have many correlated frequency amplitudes.
# PCA finds linear combinations (components) that capture most variance,
# reducing dimensionality while preserving signal structure and enabling
# denoising or interpretation of dominant modes.
n_features = X_scaled.shape[1]
pca_full = PCA(n_components=n_features)
pca_full.fit(X_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Number of components needed to explain at least 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print("=" * 50)
print("PCA RESULTS")
print("=" * 50)
print(f"Number of components for ≥95% variance: {n_components_95}")
print(f"Cumulative variance at {n_components_95} components: {cumulative_variance[n_components_95 - 1]:.4f}")

# ---------------------------------------------------------------------------
# 5. Fit PCA with selected components and reconstruct signals
# ---------------------------------------------------------------------------
pca = PCA(n_components=n_components_95)
X_transformed = pca.fit_transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_transformed)

# ---------------------------------------------------------------------------
# 6. Reconstruction error (per sample: MSE between original and reconstructed)
# ---------------------------------------------------------------------------
# Error per sample (row), then we report mean across samples
reconstruction_errors_per_sample = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
mse_reconstruction = np.mean(reconstruction_errors_per_sample)
rmse_reconstruction = np.sqrt(mse_reconstruction)

# PCA Reconstruction Accuracy (%): 100 * (1 - MSE / variance(original_data))
var_original = np.var(X_scaled)
accuracy = 100 * (1 - mse_reconstruction / var_original) if var_original != 0 else 0.0

print(f"\nReconstruction error (on standardized data):")
print(f"Mean Squared Error (MSE):  {mse_reconstruction:.6f}")
print(f"Root Mean Squared Error (RMSE):  {rmse_reconstruction:.6f}")
print(f"PCA Reconstruction Accuracy (%):  {accuracy:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# (a) Explained variance vs number of components
n_plot = min(50, n_features)  # Show first 50 components or all if fewer
ax1.bar(range(1, n_plot + 1), explained_variance_ratio[:n_plot], alpha=0.7, label="Individual")
ax1.plot(range(1, n_plot + 1), cumulative_variance[:n_plot], "r-o", markersize=4, label="Cumulative")
ax1.axhline(y=0.95, color="k", linestyle="--", alpha=0.7, label="95% threshold")
ax1.set_xlabel("Number of components")
ax1.set_ylabel("Explained variance ratio")
ax1.set_title("(a) Explained Variance vs Number of Components")
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Original vs reconstructed signal for one sample
sample_idx = 0
freq_indices = np.arange(X_scaled.shape[1])
ax2.plot(freq_indices, X_scaled[sample_idx], "b-", alpha=0.8, label="Original (scaled)")
ax2.plot(freq_indices, X_reconstructed[sample_idx], "r--", alpha=0.8, label="Reconstructed")
ax2.set_xlabel("Frequency index")
ax2.set_ylabel("Amplitude (standardized)")
ax2.set_title("(b) Original vs Reconstructed (one sample)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# (c) Distribution (histogram) of reconstruction errors
# Per-sample MSE (each sample has one reconstruction error value)
ax3.hist(reconstruction_errors_per_sample, bins=30, edgecolor="k", alpha=0.7)
ax3.set_xlabel("Reconstruction MSE (per sample)")
ax3.set_ylabel("Frequency")
ax3.set_title("(c) Distribution of Reconstruction Errors")
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("pca_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'pca_analysis.png'")

# ---------------------------------------------------------------------------
# Interpretation (comments)
# ---------------------------------------------------------------------------
# - Explained variance plot: Shows how many components capture most variance;
#   often a small number suffices (e.g. 95% in few components) → high redundancy.
# - Original vs reconstructed: Small differences mean the chosen components
#   represent the signal well; large deviations suggest lost detail or noise.
# - Histogram of errors: Concentrated near zero → good reconstruction;
#   long tail → some samples or frequencies are harder to approximate.
