"""
Autoencoder for Multivariate Signal Data (Unsupervised)
Reconstruction-based dimensionality reduction and denoising.
Uses: pandas, numpy, scikit-learn, matplotlib, tensorflow/keras.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1. Load the CSV dataset
# ---------------------------------------------------------------------------
# Unsupervised: no target column. Each row = one signal instance,
# each column = amplitude at a fixed frequency.
df = pd.read_csv("all_frequencies_1000_1.csv")
X = df.values  # Shape: (n_samples, n_features)

# ---------------------------------------------------------------------------
# 2. Split into training and testing (80:20, random_state=42)
# ---------------------------------------------------------------------------
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------------
# 3. Scale the data using StandardScaler
# ---------------------------------------------------------------------------
# Fit on training data only to avoid leakage; transform both train and test.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]
encoding_dim = 8

# ---------------------------------------------------------------------------
# 4. Build the Autoencoder
# ---------------------------------------------------------------------------
# Why Autoencoder is suitable: Unsupervised; learns a compact representation
# (encoding) and reconstructs the input. Captures nonlinear structure and
# redundancy in frequency-domain signals, useful for denoising or compression.
input_layer = keras.Input(shape=(n_features,))

# Encoder
x = layers.Dense(64, activation="relu")(input_layer)
x = layers.Dense(32, activation="relu")(x)
encoded = layers.Dense(encoding_dim, activation="relu")(x)

# Decoder
x = layers.Dense(32, activation="relu")(encoded)
x = layers.Dense(64, activation="relu")(x)
decoded = layers.Dense(n_features, activation="linear")(x)

autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)

# ---------------------------------------------------------------------------
# 5. Compile: Adam optimizer, MSE loss
# ---------------------------------------------------------------------------
autoencoder.compile(optimizer="adam", loss="mse")

# ---------------------------------------------------------------------------
# 6. Train the Autoencoder
# ---------------------------------------------------------------------------
history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
)

# ---------------------------------------------------------------------------
# 7. Reconstruct test data
# ---------------------------------------------------------------------------
X_test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)

# ---------------------------------------------------------------------------
# 8. Evaluation metrics
# ---------------------------------------------------------------------------
# MSE and RMSE between original and reconstructed test data (element-wise).
mse_reconstruction = mean_squared_error(X_test_scaled, X_test_reconstructed)
rmse_reconstruction = np.sqrt(mse_reconstruction)

# PCA-style Reconstruction Accuracy (%): 100 * (1 - MSE / variance(original_test_data))
# Why this definition: Same as PCA; measures how much of the data variance is
# preserved vs lost. High accuracy = reconstruction captures most variance.
var_test = np.var(X_test_scaled)
accuracy = 100 * (1 - mse_reconstruction / var_test) if var_test != 0 else 0.0

print("=" * 50)
print("AUTOENCODER RECONSTRUCTION METRICS")
print("=" * 50)
print(f"Mean Squared Error (MSE):  {mse_reconstruction:.6f}")
print(f"Root Mean Squared Error (RMSE):  {rmse_reconstruction:.6f}")
print(f"Reconstruction Accuracy (%):  {accuracy:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

# (a) Training loss vs Validation loss
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(history.history["loss"], label="Training loss")
ax1.plot(history.history["val_loss"], label="Validation loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
ax1.set_title("(a) Training vs Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Original vs Reconstructed signal for one test sample
ax2 = fig.add_subplot(2, 2, 2)
sample_idx = 0
feat_idx = np.arange(n_features)
ax2.plot(feat_idx, X_test_scaled[sample_idx], "b-", alpha=0.8, label="Original")
ax2.plot(feat_idx, X_test_reconstructed[sample_idx], "r--", alpha=0.8, label="Reconstructed")
ax2.set_xlabel("Feature index")
ax2.set_ylabel("Amplitude (scaled)")
ax2.set_title("(b) Original vs Reconstructed (one test sample)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# (c) Reconstruction error distribution (per-sample MSE, then histogram)
ax3 = fig.add_subplot(2, 2, 3)
# Per-sample MSE: one value per test sample
mse_per_sample = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=1)
ax3.hist(mse_per_sample, bins=30, edgecolor="k", alpha=0.7)
ax3.set_xlabel("Reconstruction MSE (per sample)")
ax3.set_ylabel("Frequency")
ax3.set_title("(c) Reconstruction Error Distribution")
ax3.grid(True, alpha=0.3, axis="y")

# (d) Per-feature reconstruction error (mean squared error per column)
ax4 = fig.add_subplot(2, 2, 4)
# MSE per feature: shape (n_features,)
mse_per_feature = np.mean((X_test_scaled - X_test_reconstructed) ** 2, axis=0)
n_bars = min(30, n_features)  # Show first 30 features or all if fewer
ax4.bar(range(n_bars), mse_per_feature[:n_bars], align="center")
ax4.set_xlabel("Feature index")
ax4.set_ylabel("MSE")
ax4.set_title("(d) Per-Feature Reconstruction Error")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("autoencoder_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'autoencoder_evaluation.png'")

# ---------------------------------------------------------------------------
# Interpretation (comments)
# ---------------------------------------------------------------------------
# (a) Loss curves: Training and validation loss decreasing = learning.
#     Validation > training often; if validation stops improving, consider
#     early stopping or smaller capacity.
# (b) Original vs Reconstructed: Close match = good reconstruction; large
#     gaps = information lost or noise not captured.
# (c) Error distribution: Concentrated near zero = most samples well
#     reconstructed; long tail = some samples harder to reconstruct.
# (d) Per-feature error: Features with high bar are harder to reconstruct
#     (e.g. high-frequency or noisy components); low bar = well captured.
