import numpy as np

# Advanced Linear Algebra
# ----------------------
print("\nAdvanced Linear Algebra Examples:")

# Eigenvalues and Eigenvectors
matrix = np.array([[4, -2], [1, 1]])
eigenvals, eigenvects = np.linalg.eig(matrix)
print("Eigenvalues:", eigenvals)
print("Eigenvectors:\n", eigenvects)

# Matrix Inverse
inv_matrix = np.linalg.inv(matrix)
print("Inverse matrix:\n", inv_matrix)

# Solving Linear Equations
# Solve for x in Ax = b
A = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(A, b)
print("Solution to linear equations:", x)

# SVD (Singular Value Decomposition)
U, s, VT = np.linalg.svd(matrix)
print("SVD components:\nU:\n", U, "\ns:", s, "\nVT:\n", VT)
