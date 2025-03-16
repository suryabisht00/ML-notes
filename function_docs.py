"""
NumPy Functions and Methods Documentation
---------------------------------------

Array Creation Methods:
---------------------
np.array(): Creates an array from a Python list or tuple
    Example: np.array([1, 2, 3]) -> Creates 1D array
    Example: np.array([[1, 2], [3, 4]]) -> Creates 2D array

np.zeros(): Creates array filled with zeros
    Syntax: np.zeros(shape, dtype=float)
    Example: np.zeros((3, 3)) -> 3x3 array of zeros

np.ones(): Creates array filled with ones
    Syntax: np.ones(shape, dtype=float)
    Example: np.ones((2, 2)) -> 2x2 array of ones

np.random.rand(): Creates array with random values [0,1]
    Example: np.random.rand(3, 3) -> 3x3 array of random values

Array Operations:
---------------
reshape(): Changes array shape while keeping data
    Syntax: array.reshape(new_shape)
    Example: array.reshape(1, 9) -> Converts to 1x9 array

transpose/T: Transposes array dimensions
    Example: array.T or array.transpose()

dot(): Matrix multiplication
    Syntax: np.dot(array1, array2)
    Example: np.dot(A, B) -> Matrix product of A and B

Statistical Operations:
--------------------
mean(): Calculates average
    Syntax: np.mean(array, axis=None)
    axis=0: column mean
    axis=1: row mean

std(): Calculates standard deviation
    Syntax: np.std(array, axis=None)
    Example: np.std(data, axis=0) -> Column-wise std

var(): Calculates variance
    Syntax: np.var(array, axis=None)
    Example: np.var(data, axis=0) -> Column-wise variance

Linear Algebra Operations:
-----------------------
linalg.eig(): Computes eigenvalues and eigenvectors
    Syntax: np.linalg.eig(square_matrix)
    Returns: (eigenvalues, eigenvectors)

linalg.inv(): Computes matrix inverse
    Syntax: np.linalg.inv(square_matrix)
    Requirement: Matrix must be non-singular

linalg.solve(): Solves linear equations Ax = b
    Syntax: np.linalg.solve(A, b)
    Example: Solves system of linear equations

Machine Learning Operations:
-------------------------
Standard Scaling: (x - mean) / std
    Used for: Feature normalization
    Property: Results in mean=0, std=1

Min-Max Scaling: (x - min) / (max - min)
    Used for: Feature scaling to [0,1] range
    Property: Preserves zero values

One-Hot Encoding: np.eye(n)[categories]
    Used for: Converting categorical variables
    Example: [0,1,2] -> [[1,0,0], [0,1,0], [0,0,1]]
"""

# Example usage demonstrations
if __name__ == "__main__":
    import numpy as np
    
    # Array creation examples
    print("\nArray Creation Examples:")
    print(np.array([1, 2, 3]))
    print(np.zeros((2, 2)))
    
    # Statistical operations
    data = np.array([[1, 2], [3, 4]])
    print("\nStatistical Operations:")
    print("Mean:", np.mean(data, axis=0))  # Column means
    print("Std:", np.std(data, axis=1))    # Row standard deviations
