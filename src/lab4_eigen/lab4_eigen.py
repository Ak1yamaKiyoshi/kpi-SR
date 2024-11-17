import numpy as np
from lib_print import *
np.set_printoptions(suppress=True)


__default_print_format = ANSI.FG.BRIGHT_BLACK + ANSI.Styles.ITALIC
__default_numbers_format = "10.5f"
__default_text_format = ANSI.Styles.BOLD 

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
# підставити значення з маткаду 


print(printer(
    a, 
    f"{__default_text_format} Starting matrix: ",default_style= __default_print_format, formatting=__default_numbers_format
) + "\n")

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
        -mat_a_i[idx,j]/mat_a_i[idx,idx-1] # що тут вбіса відбувається 
        if j != m - i - 1 else 1 / mat_a_i[idx, idx-1] 
    for j in range(m)]
    mat_m[idx-1] = row

    mat_a_i = np.linalg.inv(mat_m) @ mat_a_i @ mat_m

    
    if s is None:
        s = mat_m.copy()
    else: s = s @ mat_m

    print(printer(
        mat_m,
        f"{__default_text_format} Matrix M{idx}", default_style= __default_print_format, formatting=__default_numbers_format
    ) + "\n")
    print(printer(
        mat_a_i, 
        f"{__default_text_format} Matrix A{i}",default_style= __default_print_format, formatting=__default_numbers_format
    ) + "\n")

p = mat_a_i.copy()

print(printer(
    p, f"{__default_text_format} Matrix P, phronebius form: ", default_style= __default_print_format, formatting=__default_numbers_format
) + "\n") 
print(printer(
    s, f"{__default_text_format} Matrix S: ", default_style= __default_print_format, formatting=__default_numbers_format
) + "\n") 

# що таке форма фронебіуса
# що таке eigenvalues (власні значення)
v = np.array([-p[0,3], -p[0,2], -p[0,1], -p[0,0], 1])
lambda_values = np.roots(v[::-1]) # eigenvals 
# що робить метод roots, 
# як знаходяться власні значення з нормальної форми фронебіуса 

print(printer(
    lambda_values.reshape(-1, 1)[:4], f"{__default_text_format} Eigenvalues ", default_style= __default_print_format, formatting=__default_numbers_format, 
) + "\n") 


# y
mat_y = np.array([
    [l**3, l**2, l, 1] for l in lambda_values
])
# чому значення для y рахуються як eigenvalues у якійсь степені? 
print(printer(
    mat_y, f"{__default_text_format} Y: ", default_style= __default_print_format, formatting=__default_numbers_format, 
) + "\n") 

# що таке eigenvectors (власні вектори)
# eigenvec 
eigenvec = s @ mat_y.T
print(printer(
    eigenvec, f"{__default_text_format} Eigenvectors: ", default_style= __default_print_format, formatting=__default_numbers_format, 
) + "\n") 


np_eigenvalues, np_eigenvectors = np.linalg.eig(a)
print(printer(
    np_eigenvalues.reshape(-1, 1)[:4], f"{__default_text_format} (numpy) Eigenvalues ", default_style= __default_print_format, formatting=__default_numbers_format, 
) + "\n") 


# чому eigenvalues через numpy у іншому порядку?  
errors = []
for i, (lambda_i, v) in enumerate(zip(lambda_values, eigenvec.T), 1):
    errors.append(np.linalg.norm(a @ v - lambda_i * v))

print(printer(
     np.array(errors).reshape(-1, 1)[:4], f"{__default_text_format} (numpy) eigenvectorc - (own) eigenvectors ", default_style= __default_print_format, formatting=__default_numbers_format, 
) + "\n") 

