import numpy as np

# Data Preprocessing for Machine Learning
# ------------------------------------
"""
Key Preprocessing Functions Explained:

1. Standardization (Z-score normalization):
   - Purpose: Scales features to have mean=0 and std=1
   - Formula: z = (x - μ) / σ
   - Use Case: When data follows normal distribution
   - Benefits: Helps in gradient descent convergence
"""
# Generate sample data
data = np.random.randn(100, 4)  # 100 samples, 4 features

# Standardization (Z-score normalization)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
standardized = (data - mean) / std
print("\nStandardized data statistics:")
print("Mean:", np.mean(standardized, axis=0))
print("Std:", np.std(standardized, axis=0))

"""
2. Min-Max Scaling:
   - Purpose: Scales features to fixed range [0,1]
   - Formula: x_scaled = (x - x_min) / (x_max - x_min)
   - Use Case: When you need bounded values
   - Benefits: Preserves zero values and sparsity
"""
# Min-Max Scaling
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
normalized = (data - min_vals) / (max_vals - min_vals)
print("\nNormalized data range:")
print("Min:", np.min(normalized, axis=0))
print("Max:", np.max(normalized, axis=0))

"""
3. One-Hot Encoding:
   - Purpose: Converts categorical variables to binary vectors
   - Method: Creates binary columns for each category
   - Use Case: Converting categorical data for ML models
   - Example: category 2 in 4 classes → [0,0,1,0]
"""
# One-hot encoding simulation
categories = np.array([0, 1, 2, 1, 0])
one_hot = np.eye(3)[categories]
print("\nOne-hot encoded data:\n", one_hot)

# Additional preprocessing methods
# Handling Missing Values
def handle_missing(data):
    """Replace missing values with mean of column"""
    return np.nan_to_fill(data, np.mean(data, axis=0))

# Feature Scaling with Robust Scaler (handles outliers)
def robust_scale(data):
    """Scale features using statistics that are robust to outliers"""
    median = np.median(data, axis=0)
    q75, q25 = np.percentile(data, [75, 25], axis=0)
    iqr = q75 - q25
    return (data - median) / iqr
