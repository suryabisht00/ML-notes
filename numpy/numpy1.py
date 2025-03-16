import numpy as np

# NumPy Basics
# ------------
# NumPy is a fundamental package for scientific computing in Python
# Main object is the ndarray (N-dimensional array)

# Creating Arrays
# --------------
array1 = np.array([1, 2, 3, 4])  # 1D array
array2 = np.array([[1, 2], [3, 4]])  # 2D array
zeros = np.zeros((3, 3))  # Create 3x3 array of zeros
ones = np.ones((2, 2))  # Create 2x2 array of ones
random_array = np.random.rand(3, 3)  # Create 3x3 array of random values

print("\nArray Creation Examples:")
print("1D array:", array1)
print("2D array:\n", array2)
print("Zeros array:\n", zeros)
print("Ones array:\n", ones)
print("Random array:\n", random_array)

# Array Operations
# ---------------
# NumPy operations are vectorized (element-wise)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
sum_array = a + b  # Addition
product_array = a * b  # Multiplication
squared = a ** 2  # Exponentiation

print("\nArray Operations:")
print("Sum of arrays:", sum_array)
print("Product of arrays:", product_array)
print("Squared array:", squared)

# Array Indexing and Slicing
# -------------------------
sample = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
first_row = sample[0]  # Get first row
element = sample[1, 2]  # Get element at row 1, column 2
subset = sample[0:2, 1:]  # Get subset of rows 0-1 and columns 1 onwards

print("\nIndexing and Slicing:")
print("First row:", first_row)
print("Element at (1,2):", element)
print("Subset:\n", subset)

# Common Functions
# ---------------
mean_value = np.mean(sample)  # Calculate mean
sum_value = np.sum(sample)  # Calculate sum
max_value = np.max(sample)  # Find maximum value
min_value = np.min(sample)  # Find minimum value

print("\nCommon Functions:")
print("Mean value:", mean_value)
print("Sum value:", sum_value)
print("Maximum value:", max_value)
print("Minimum value:", min_value)

# Array Reshaping
# --------------
reshaped = sample.reshape(1, 9)  # Reshape to 1x9 array
transposed = sample.T  # Transpose array

print("\nReshaping:")
print("Reshaped array:", reshaped)
print("Transposed array:\n", transposed)

# Broadcasting
# -----------
# NumPy can operate on arrays of different shapes
scalar = 2
scaled_array = sample * scalar  # Multiply each element by 2

print("\nBroadcasting:")
print("Scaled array:\n", scaled_array)

# Linear Algebra
# -------------
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
dot_product = np.dot(matrix_a, matrix_b)  # Matrix multiplication
determinant = np.linalg.det(matrix_a)  # Calculate determinant

print("\nLinear Algebra:")
print("Dot product:\n", dot_product)
print("Determinant:", determinant)

# Statistical Functions
# -------------------
std_dev = np.std(sample)  # Standard deviation
variance = np.var(sample)  # Variance
median = np.median(sample)  # Median value

print("\nStatistical Functions:")
print("Standard deviation:", std_dev)
print("Variance:", variance)
print("Median:", median)
