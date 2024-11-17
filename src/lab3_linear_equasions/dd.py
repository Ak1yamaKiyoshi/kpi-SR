import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Початкова система
matrix_a = np.array([
    [2.12, 0.42, 1.34, 0.88],
    [0.42, 3.95, 1.87, 0.43],
    [1.34, 1.87, 2.98, 0.46],
    [0.88, 0.43, 0.46, 4.44]
])

vector_b = np.array([11.172, 0.115, 0.009, 9.349])


print("---")
print("Крок 1: Початкова система")
print("Матриця A:")
print(matrix_a)
print("\nВектор b:")
print(vector_b)


print("matrix")
print(f"віднімаємо від третього рядку половину першого") 
multiplier = 1/2
print(f"{vector_b[2]} - {multiplier * vector_b[0]}")
print(f"{matrix_a[2]} - {multiplier * matrix_a[0]} = {matrix_a[2] - multiplier * matrix_a[0]}")
matrix_a[2] = matrix_a[2] - multiplier * matrix_a[0]
vector_b[2] = vector_b[2] - multiplier * vector_b[0]

print("Після перетворення:")
print(matrix_a)
print("\nВектор b:")
print(vector_b)



print(f"Від першого рядку віднімаємо 2/3 третього аби позубутись досягти діагональної переваги у рядку  ")
multiplier = 2/3
matrix_a[0] = matrix_a[0] - multiplier * matrix_a[2]
vector_b[0] = vector_b[0] - multiplier * vector_b[2]

print("Після перетворення:")
print(matrix_a)
print("\nВектор b:")
print(vector_b)

print("") 
multiplier = 1/6
matrix_a[0] = matrix_a[0] - multiplier * matrix_a[3]
vector_b[0] = vector_b[0] - multiplier * vector_b[3]

print("\nФінальна матриця:")
print(matrix_a)
print("\nФінальний вектор b:")
print(vector_b)

print("\nКрок 6: Перевіряємо фінальне діагональне переважання:")
for i in range(4):
    diagonal = abs(matrix_a[i,i])
    row_sum = sum(abs(matrix_a[i,j]) for j in range(4) if j != i)
    print(f"\nРядок {i+1}:")
    print(f"|{diagonal:.3f}| > {row_sum:.3f}")
    print(f"Є переважання? {diagonal > row_sum}")