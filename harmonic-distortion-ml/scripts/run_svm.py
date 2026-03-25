"""
Support Vector Regression (SVR) - Training and Evaluation Script
Trains an SVR model on frequency-based data with feature engineering.
Uses only: pandas, numpy, scikit-learn, matplotlib.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load the dataset (CSV format)
# ---------------------------------------------------------------------------
# First column = frequency, last column = target (continuous output)
df = pd.read_csv("all_frequencies_1000_1.csv")

# ---------------------------------------------------------------------------
# 2. Extract frequency (first column) and target (last column)
# ---------------------------------------------------------------------------
frequency = df.iloc[:, 0].values  # First column: frequency
y = df.iloc[:, -1].values        # Last column: target/output

# ---------------------------------------------------------------------------
# 3. Feature engineering on the frequency column
# ---------------------------------------------------------------------------
# Why: The relationship may be nonlinear/sinusoidal (e.g. physics-based).
# sin/cos capture periodic behavior; f² captures quadratic effects;
# log(f+1e-6) captures logarithmic scaling often seen in frequency response.
f = frequency.astype(float)
X_engineered = np.column_stack([
    np.sin(2 * np.pi * f),      # sin(2πf) - periodic component
    np.cos(2 * np.pi * f),      # cos(2πf) - periodic component
    f ** 2,                     # f² - quadratic dependence
    np.log(f + 1e-6),           # log(f + 1e-6) - avoid log(0)
])
# Use column names for clarity (optional for later)
feature_names = ["sin(2πf)", "cos(2πf)", "f²", "log(f+1e-6)"]
X = pd.DataFrame(X_engineered, columns=feature_names)

# ---------------------------------------------------------------------------
# 4. Split data into training and testing (80:20, random_state=42)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 5. Scale both input features and output using StandardScaler
# ---------------------------------------------------------------------------
# SVR is sensitive to feature scale; scaling y helps the loss (epsilon-tube).
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
# Reshape y for scaler (expects 2D)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_actual = y_test  # Keep original for evaluation

# ---------------------------------------------------------------------------
# 6. Train the SVR model
# ---------------------------------------------------------------------------
# Why SVR: Handles nonlinear relationships via RBF kernel, robust to outliers,
# and works well for smooth regression (e.g. sinusoidal/physics-based).
model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)
model.fit(X_train_scaled, y_train_scaled)

# ---------------------------------------------------------------------------
# 7. Predict and inverse-transform scaled output to original scale
# ---------------------------------------------------------------------------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# ---------------------------------------------------------------------------
# 8. Compute evaluation metrics (on original scale)
# ---------------------------------------------------------------------------
mae = mean_absolute_error(y_test_actual, y_pred)
mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)
mean_y_test = np.mean(y_test_actual)
accuracy = 100 * (1 - mae / mean_y_test) if mean_y_test != 0 else 0.0

# ---------------------------------------------------------------------------
# 9. Print evaluation metrics with labels
# ---------------------------------------------------------------------------
print("=" * 50)
print("EVALUATION METRICS (SVR)")
print("=" * 50)
print(f"Mean Absolute Error (MAE):  {mae:.6f}")
print(f"Mean Squared Error (MSE):  {mse:.6f}")
print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
print(f"R² Score:  {r2:.6f}")
print(f"Regression Accuracy (%):  {accuracy:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 10. Plots using matplotlib
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

# (a) Actual vs Predicted with y = x reference line
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(y_test_actual, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
min_val = min(y_test_actual.min(), y_pred.min())
max_val = max(y_test_actual.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
ax1.set_xlabel("Actual values")
ax1.set_ylabel("Predicted values")
ax1.set_title("(a) Actual vs Predicted")
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Residuals vs Predicted values
ax2 = fig.add_subplot(2, 2, 2)
residuals = y_test_actual - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
ax2.axhline(y=0, color="r", linestyle="--", lw=2)
ax2.set_xlabel("Predicted values")
ax2.set_ylabel("Residuals")
ax2.set_title("(b) Residuals vs Predicted")
ax2.grid(True, alpha=0.3)

# (c) Line plot: Actual vs Predicted for first 100 test samples
ax3 = fig.add_subplot(2, 2, 3)
n_show = min(100, len(y_test_actual))
x_range = np.arange(n_show)
ax3.plot(x_range, y_test_actual[:n_show], "b-", label="Actual", alpha=0.8)
ax3.plot(x_range, y_pred[:n_show], "r--", label="Predicted", alpha=0.8)
ax3.set_xlabel("Sample index")
ax3.set_ylabel("Output value")
ax3.set_title("(c) Actual vs Predicted (first 100 test samples)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# (d) Histogram of residuals (prediction errors)
ax4 = fig.add_subplot(2, 2, 4)
ax4.hist(residuals, bins=30, edgecolor="k", alpha=0.7)
ax4.axvline(x=0, color="r", linestyle="--", lw=2)
ax4.set_xlabel("Residual")
ax4.set_ylabel("Frequency")
ax4.set_title("(d) Histogram of Residuals")
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("svr_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'svr_evaluation.png'")
