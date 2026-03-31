import numpy as np

# Example: r x n array
r = 5
n = 3
matrix = np.random.rand(r, n)
print(matrix)

new_rows = r*2  # c*3

# Old and new row positions
old_indices = np.linspace(0, 1, r)
new_indices = np.linspace(0, 1, new_rows)

# Interpolate each column
new_matrix = np.array([
    np.interp(new_indices, old_indices, matrix[:, col])
    for col in range(n)
]).T

print(new_matrix.shape)  # (12, 3)
print(new_matrix)

from scipy.interpolate import interp1d

f = interp1d(old_indices, matrix, axis=0, kind='cubic')
print(f)
new_matrix = f(new_indices)
print(new_matrix)

print(old_indices)
print(new_indices)