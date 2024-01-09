import numpy as np

# Example: Finding x and y indices of the maximum value in a 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 9, 5]])

# Find the flat index of the maximum value in the matrix
flat_index_max = np.argmax(matrix)

# Convert the flat index to x and y indices using np.unravel_index
x_index, y_index = np.unravel_index(flat_index_max, matrix.shape)

print("Flat index of the maximum value:", flat_index_max)
print("X (row) index of the maximum value:", x_index)
print("Y (column) index of the maximum value:", y_index)
print("Maximum value in the matrix:", matrix[x_index, y_index])
