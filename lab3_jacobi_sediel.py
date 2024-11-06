from drawlib import *
import numpy as np
np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

def is_strictly_diagonally_dominant(matrix):
    for i in range(matrix.shape[0]):
        if abs(matrix[i, i]) <= np.sum(np.abs(matrix[i, :])) - abs(matrix[i, i]):
            return False
    return True


vector_b = np.array([11.172, 0.115, 0.009, 9.349]).T
matrix_a = np.array([
  [2.12, 0.42, 1.34, 0.88],
  [0.42, 3.95, 1.87, 0.43],
  [1.34, 1.87, 2.98, 0.46],
  [0.88, 0.43, 0.46, 4.44],
])

# Diagonal dominant
matrix_a = np.array([[5.79, 0.42, 1.34, 0.88],
  [0.42, 7.62, 1.87, 0.43],
  [1.34, 1.87, 6.65, 0.46],
  [0.88, 0.43, 0.46, 8.11]])

a = matrix_a.copy()
b = vector_b.copy()
n = a.shape[0] 


def jacobi(a, b, epsilon=1e-5, max_iterations=1000):
    n = a.shape[0]
    D_inv = np.diag(1 / np.diag(a))
    iteration_matrix = np.eye(n) - D_inv @ a
    iteration_vector = D_inv @ b
    x = np.zeros_like(b)

    for _ in range(max_iterations):
        x_new = iteration_matrix @ x + iteration_vector
        if np.max(np.abs(x_new - x)) < epsilon:
            return x_new
        x = x_new
    return x 


def zeidel(a, b):
    n = a.shape[0]
    x = np.zeros_like(b)
    epsilon = 1e-6
    max_iterations = 1000
    
    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            sum1 = sum(a[i,j] * x_new[j] for j in range(i))
            sum2 = sum(a[i,j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - sum1 - sum2) / a[i,i]
        # непанятна 
        if np.max(np.abs(x_new - x)) < epsilon:
            break
        x = x_new
    
    return x

print("jacobi: ")
print(a@jacobi(a, b)-b)
print("zeidel:")
print(a@zeidel(a, b)-b)