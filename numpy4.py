import numpy as np

# Advanced Statistical Operations
# -----------------------------
print("\nAdvanced Statistical Operations:")

# Generate random data
data = np.random.normal(100, 20, 1000)

# Advanced Statistics
percentiles = np.percentile(data, [25, 50, 75])
print("Quartiles:", percentiles)

# Correlation and Covariance
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5
correlation = np.corrcoef(x, y)
covariance = np.cov(x, y)
print("Correlation matrix:\n", correlation)
print("Covariance matrix:\n", covariance)

# Histogram computation
hist, bins = np.histogram(data, bins=30)
print("Histogram counts:", hist)
print("Histogram bins:", bins)
