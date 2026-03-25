"""
Random Forest Regression on Multivariate Frequency-Domain Signal Data
Supervised formulation: predict last frequency from first (n-1) frequencies.
Uses only: pandas, numpy, scikit-learn, matplotlib.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load the CSV dataset
# ---------------------------------------------------------------------------
# Each row = one signal instance, each column = amplitude at a fixed frequency.
# Why the original RF approach was invalid: If we treated the whole row as
# "features" with a separate external target, we'd mix regression with
# unsupervised structure. If we used all columns as features with no clear
# target, there is no supervised task. Here we define a clear supervised
# task: predict one frequency from the others (reconstruction-based).
df = pd.read_csv("all_signals_1000_1.csv")

# ---------------------------------------------------------------------------
# 2. Formulate supervised task: X = first (n-1) frequency columns, y = last
# ---------------------------------------------------------------------------
# Why this formulation is correct: We have only frequency amplitudes (no
# separate target column). Using the last frequency as y and the rest as X
# gives a well-defined regression: "given amplitudes at frequencies 1..n-1,
# predict amplitude at frequency n." This is reconstruction-based and
# interpretable (which frequencies drive the last one).
n_cols = df.shape[1]
X = df.iloc[:, :-1]   # First (n-1) frequency columns as input features
y = df.iloc[:, -1]    # Last frequency column as target

# ---------------------------------------------------------------------------
# 3. Split data into training and testing (80:20, random_state=42)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Train the Random Forest Regressor
# ---------------------------------------------------------------------------
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 5. Predict and evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mean_y_test = np.mean(y_test)
accuracy = 100 * (1 - mae / mean_y_test) if mean_y_test != 0 else 0.0

# ---------------------------------------------------------------------------
# 6. Print evaluation metrics
# ---------------------------------------------------------------------------
print("=" * 50)
print("EVALUATION METRICS (Random Forest - Frequency Data)")
print("=" * 50)
print(f"Mean Absolute Error (MAE):  {mae:.6f}")
print(f"Mean Squared Error (MSE):  {mse:.6f}")
print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
print(f"R² Score:  {r2:.6f}")
print(f"Regression Accuracy (%):  {accuracy:.2f}")
print("=" * 50)

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))

# (a) Actual vs Predicted values for the target frequency
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
ax1.set_xlabel("Actual (target frequency)")
ax1.set_ylabel("Predicted")
ax1.set_title("(a) Actual vs Predicted (target frequency)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Residuals vs Predicted values
ax2 = fig.add_subplot(2, 2, 2)
residuals = y_test.values - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
ax2.axhline(y=0, color="r", linestyle="--", lw=2)
ax2.set_xlabel("Predicted values")
ax2.set_ylabel("Residuals")
ax2.set_title("(b) Residuals vs Predicted")
ax2.grid(True, alpha=0.3)

# (c) Feature importance: influence of input frequencies
# Interpretation: High importance = that frequency amplitude strongly drives
# the prediction of the last frequency; low importance = weak or redundant.
ax3 = fig.add_subplot(2, 2, 3)
importance = model.feature_importances_
feature_names = X.columns.tolist()
n_show = min(20, len(feature_names))  # Show top 20 or all if fewer
indices = np.argsort(importance)[::-1][:n_show]
top_imp = importance[indices]
top_names = [str(feature_names[i]) for i in indices]
ax3.barh(range(n_show), top_imp[::-1], align="center")
ax3.set_yticks(range(n_show))
ax3.set_yticklabels(top_names[::-1])
ax3.set_xlabel("Feature importance")
ax3.set_title("(c) Feature importance (input frequencies)")
ax3.grid(True, alpha=0.3, axis="x")

# (d) Line plot: actual vs predicted for first 100 test samples
ax4 = fig.add_subplot(2, 2, 4)
n_show_samples = min(100, len(y_test))
x_range = np.arange(n_show_samples)
ax4.plot(x_range, y_test.values[:n_show_samples], "b-", label="Actual", alpha=0.8)
ax4.plot(x_range, y_pred[:n_show_samples], "r--", label="Predicted", alpha=0.8)
ax4.set_xlabel("Test sample index")
ax4.set_ylabel("Target frequency amplitude")
ax4.set_title("(d) Actual vs Predicted (first 100 test samples)")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rf_frequency_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nPlot saved as 'rf_frequency_evaluation.png'")
