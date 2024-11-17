import numpy as np
from scipy.special import factorial
from scipy import integrate
import sympy as sp
import time

def nth_prime_sp(expr, var, n):
    d_expr = sp.diff(expr, var, n)
    return sp.lambdify([var], d_expr, modules='numpy'), d_expr


def gaussian(f, f2m_prime, a, b, m):
    t_start = time.time()
    
    z, w = np.polynomial.legendre.leggauss(m)
    x = 0.5 * ((b - a) * z + b + a)
    w = 0.5 * (b - a) * w
    integral = np.sum(w * f(x)),
 
 
    t_end = time.time()
    max_y2m = np.max(np.abs(f2m_prime(x)))
    error = ((factorial(m)**4 * (b - a)**(2*m+1)) / 
            ((2*m+1) * factorial(2*m)**3)) * max_y2m

    return {
        "value": float(integral[0]),
        "analytical_error": abs(float(error)),
        "execution_time":  t_end - t_start
    }


def trapezoidal(f, f_prime2, a, b, n):
    t_start = time.time()
    
    x = np.linspace(b, a, n+1)
    y = f(x)
    h = (b - a) / n
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    t_end = time.time()
    error = ((b - a)**3 / (12 * n**2)) * np.max(np.abs(f_prime2(x)))
    return {
        "analytical_error": abs(float(error)),
        "value": float(integral), 
        "execution_time":  t_end - t_start}

x_sympy = sp.Symbol("x")
f_sympy = sp.cos(x_sympy) / (x_sympy + 1)
f = lambda x: np.cos(x) / (x + 1)
f_prime2 = lambda x: (2 * ((x + 1) * np.sin(x) + np.cos(x)))\
                            / (x + 1)**3 - np.cos(x) / (x + 1)


a, b = 0.8, 10.07
tolerance = 0.0001
# tolerance = 0.0000001

f2m_prime = None
final_m = 0
final_n = 0

print(f"Tolerance: {tolerance:1.10f}")

for m_temp in range(1, 100):
    f2m_prime = nth_prime_sp(f_sympy, x_sympy, 2*m_temp)[0]
    res = gaussian(f, f2m_prime, a, b, m_temp)    
    error = res['analytical_error']
    if error < tolerance:
        final_m = m_temp
        print(f"m for gaussian quadrature:      {m_temp}, error: {error:1.15f}")
        break

for n in range(1, 1000):
    res = trapezoidal(f, f_prime2, a, b, n)
    error = res['analytical_error']
    if error < tolerance:
        final_n = n
        print(f"n steps for trapezoidal method: {n}, error: {error:1.15f}")
        break


res_gaussian = gaussian(f, f2m_prime, a, b, m_temp)
res_trapezoidal = trapezoidal(f, f_prime2, a, b, final_n)
sympy_sol = integrate.quad(f, a, b)[0]



diff_gauss = abs(sympy_sol - res_gaussian['value'])
diff_trap = abs(sympy_sol - res_trapezoidal['value'])


print(f"""

Result: 
  * gaussian quadrature:  {res_gaussian['value']:+1.10f}
  * trapezoidal method:   {res_trapezoidal['value']:+1.10f}
  * sympy integrate quad: {sympy_sol:+1.10f}
 
Errors: 
  * |sympy - gaussian|:   {diff_gauss:+1.10f}
  *  gaussian error:      {res_gaussian['analytical_error']:+1.10f}

  * |sympy - trapezoidal|:{diff_trap:+1.10f}
  *  trapezoidal error:   {res_trapezoidal['analytical_error']:+1.10f}
""")


print(f"""
gaussian time:    {res_gaussian['execution_time']:1.10f}s.
trapezoidal time: {res_trapezoidal['execution_time']:1.10f}s.
""")

""" 
Метод трапецій потребує більшу кількість точок; (2 кроки проти 3);
Метод гаусса точніший. (+0.0000001614 проти +0.0000537052);


У ході виконання лабораторної роботи я дізнався про методи чисельного інтегрування функцій, а саме метод трапецій та квадратурну формулу Гауса.
Для заданої функції метод трапецій та метод Гауса мають різну кількість кроків для досягнення потрібної точності - методу трапецій знадобилось 3 кроки, тоді як методу Гауса лише 2. Метод Гауса виявився не тільки ефективнішим за кількістю кроків, але й значно точнішим - його похибка становить +0.0000001614, тоді як похибка методу трапецій +0.0000537052.
Обидві отримані похибки є меншими за задану tolerance = 0.0001, що підтверджує коректність обчислень. 
Оскільки остаточне значення визначеного інтегралу обома розглянутими методами зійшлось в межах заданої похибки, можна зробити висновок, що всі обчислення, включаючи заміну змінної, було виконано правильно. Метод Гауса продемонстрував кращі результати як за кількістю необхідних кроків, так і за точністю обчислень.
"""

