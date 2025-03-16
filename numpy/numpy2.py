import numpy as np

# Advanced Array Operations
# -----------------------
print("\nAdvanced Array Operations Examples:")

# Array Stacking
horizontal = np.hstack((np.array([1,2,3]), np.array([4,5,6])))
vertical = np.vstack((np.array([1,2,3]), np.array([4,5,6])))
print("Horizontal stack:", horizontal)
print("Vertical stack:", vertical)

# Array Splitting
array = np.array([1,2,3,4,5,6])
split = np.array_split(array, 3)
print("Split array:", split)

# Universal Functions (ufuncs)
angles = np.array([0, 30, 45, 60, 90])
sin_values = np.sin(np.deg2rad(angles))
print("Sine values:", sin_values)

# Fancy Indexing
arr = np.arange(12).reshape(3,4)
fancy_idx = arr[[0,1,2], [0,2,3]]
print("Fancy indexing result:", fancy_idx)
