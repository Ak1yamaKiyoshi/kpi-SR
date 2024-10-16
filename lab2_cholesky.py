import numpy as np


vector_b = np.array([0.3, 0.5, 0.7, 0.9]).T
matrix_a = np.array([
    [1.00, 0.42, 0.54, 0.66],
    [0.42, 1.00, 0.32, 0.44],
    [0.54, 0.32, 1.00, 0.22],
    [0.66, 0.44, 0.22, 1.00]
])

a = matrix_a.copy()
b = vector_b.copy()
n = a.shape[0]

u = np.zeros_like(a)

for i in range(n):
    for j in range(i, n):
        if i == j:
            sum = np.sum([u[k, i]**2 for k in range(i)])
            u[i, i] = np.sqrt(a[i, i] - sum)
        else:
            sum = np.sum([u[k, i] * u[k, j] for k in range(i)])
            u[i, j] = (a[i, j] - sum) / u[i, i]

# T'y = b
y = np.zeros_like(b)
for i in range(n):
    sum = np.sum([u[k, i] * y[k] for k in range(i)])
    y[i] = (b[i] - sum) / u[i, i]

# Tx = y
x = np.zeros_like(b)
for i in range(n-1, -1, -1):
    sum = np.sum([u[i, k] * x[k] for k in range(i+1, n)])
    x[i] = (y[i] - sum) / u[i, i]

print("Is Ax = b? ", np.allclose(b, a @ x))
print("Solution x:", x)

print("A - U.T @ U (should be close to zero):")
print(a - u.T @ u)

print("\nOriginal b:")
print(b)

print("\n(should be close to b):")
print("A @ x", a @ x)
print("b    ",b)
