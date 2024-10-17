import numpy as np
np.set_printoptions(suppress=True)


def default_matrix():
    t = 7.0
    k = 3 * (3 - 4) + 1
    a = 0.11 * t
    b = 0.02 * k
    g = 0.02 * k
    d = 0.015 * t
    return np.array([
        [6.26 + a, 1.10 - b, 0.97 + g, 1.24 - d],
        [1.10 - b, 4.16 - a, 1.30, 0.16],
        [0.97 + g, 1.30, 5.44 + a, 2.1],
        [1.24 - d, 0.16, 2.10, 6.10 - a]
    ]), 4
    
a, m = default_matrix()
true_eigenvalues, true_eigenvectors = np.linalg.eig(a)

""" 
1. Порахувати нормальну форму фронебіуса (скрипт)
 * Показати M_i^-1, M_i, результуючу матрицю P.
   Порахувати характеристичне рівняння методом данилевського (скрипт)
3. Знайти власні числа характеристичного рівняння (маткад) -> (numpy)
4. Знайти власнтй вектор для кожного власного числа (скрипт)
5. Перевірити точність підстановкою у рівняння знайдені числа та вектори. 
    A * x = lambda * x
"""

# Нормальна форма фронебіуса 
mat_a_i = a.copy()
s = None
for i in range(1, m):
    idx = m - i
    mat_m = np.eye(m)
    
    row = [ 
        -mat_a_i[idx,j]/mat_a_i[idx,idx-1]
        if j != m - i - 1 else 1 / mat_a_i[idx, idx-1]
    for j in range(m)]
    mat_m[idx-1] = row

    mat_a_i = np.linalg.inv(mat_m) @ mat_a_i @ mat_m
    
    if s is None:
        s = mat_m.copy()
    else: s = s @ mat_m
    print(f"M{idx}\n{mat_m}\nA{i}\n{mat_a_i}\n\n")
p = mat_a_i.copy()

print(f"P:\n{p}\n\n")

# eigenval
v = np.array([-p[0,3], -p[0,2], -p[0,1], -p[0,0], 1])
lambda_values = np.roots(v[::-1]) # eigenvals 

print(" > Eigenvalues\n")
for i, l in enumerate(lambda_values):
    print(f"λ_{i} = {l}")

# y
mat_y = np.array([
    [l**3, l**2, l, 1] for l in lambda_values
])

print(" > Y\n")
for i, l in enumerate(mat_y):
    print(f"y_{i} = {l}")

# eigenvec 
print(" > Eigenvectors\n")
eigenvec = s @ mat_y.T
for i, l in enumerate(eigenvec.T, 1):
    print(f"v_{i} = {l}")

print("\nВласні значення за допомогою numpy:")
np_eigenvalues, np_eigenvectors = np.linalg.eig(a)
for i, ev in enumerate(np_eigenvalues, 1):
    print(f"λ_{i} = {ev:.5f}")

print("\nПеревірка точності:")
for i, (lambda_i, v) in enumerate(zip(lambda_values, eigenvec.T), 1):
    error = np.linalg.norm(a @ v - lambda_i * v)
    print(f"Помилка для λ_{i}: {error:.2e}")
