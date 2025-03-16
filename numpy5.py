import numpy as np

# Advanced Broadcasting and Matrix Manipulations
# -------------------------------------------
print("\nAdvanced Broadcasting and Matrix Operations:")

# Complex Broadcasting
grid = np.zeros((4, 4))
x = np.array([1, 2, 3, 4])
y = np.array([[1], [2], [3], [4]])
broadcast_sum = x + y
print("Broadcasting result:\n", broadcast_sum)

# Advanced Matrix Operations
matrix = np.array([[1,2,3], [4,5,6], [7,8,9]])
# Trace
print("Matrix trace:", np.trace(matrix))
# Matrix power
print("Matrix power:\n", np.linalg.matrix_power(matrix, 2))
# Matrix rank
print("Matrix rank:", np.linalg.matrix_rank(matrix))

# Complex number operations
complex_array = np.array([1+2j, 3-4j, -1+0j])
print("Complex array magnitude:", np.abs(complex_array))
print("Complex array angle:", np.angle(complex_array))
