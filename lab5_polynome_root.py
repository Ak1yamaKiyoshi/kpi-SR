import numpy as np 
import matplotlib.pyplot as plt


def f(x, k=0, alpha=1):
    a0, a1, a2, a3, a4, a5 = 0, -2, 1, 5, -2, 1
    return a5*(1+alpha)*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0*k 


def bisection(f, a=-10000, b=+10000, eps=1e-5): 
    c = 0.0
    while (b-a) / 2 > eps:
        c = (a + b) / 2 
        if f(c) * f(b) > 0: # а чому множення 
            b = c 
        else: a = c
    return c


def chord(f, a=-10, b=+10, eps=1e-5):
    c = 1.0
    while  abs(f(c)) > eps:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a)) # шо це таке
        if f(c) * f(b) > 0:
            b = c 
        else: a = c
    return c


def derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x)) / eps

def newton_method(f, x0, eps=1e-5, max_iter=1000):
    x = x0
    x_new = x + 1
    while abs(x_new - x) > eps:
        x = x_new
        fx = f(x)
        f_prime_x = derivative(f, x, eps=eps) 
        x_new = x - fx / f_prime_x # чому це працює? прорахувати приклад 
    return x



a, b = -2, 2
bisection_root = bisection(f, a, b)
chord_root = chord(f, a, b)
newton_root = newton_method(f, b)

coeffs = [1, -2, 5, 1, -2, 0]
np_roots = np.roots(coeffs)
np_roots = [root.real for root in np_roots if root.imag == 0]

x_vals = np.linspace(a, b, 1000)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0, color="black", linestyle="--")
plt.scatter(bisection_root, f(bisection_root), color="red", label="Bisection Root", zorder=5)
plt.scatter(chord_root, f(chord_root), color="green", label="Chord Root", zorder=5)
plt.scatter(newton_root, f(newton_root), color="blue", label="Newton Root", zorder=5)
for np_root in np_roots:
    plt.scatter(np_root, f(np_root), color="purple", marker="x", label="Numpy Root", zorder=5)

plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Roots of the Function Found by Different Methods")
plt.show()


print(f"""
Range: {a}, {b}
Bisection method root: {bisection_root}
Chord method root:     {chord_root}
Newton method root:    {newton_root}
Numpy roots:           {np_roots}
""")