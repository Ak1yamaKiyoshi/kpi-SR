import numpy as np 

def f(x, k=0, alpha=1):
    a0, a1, a2, a3, a4, a5 = 0, -2, 1, 5, -2, 1
    return a5*(1+alpha)*x**5 + a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0*k 

def bisection(f, a=-10, b=+10, eps=1e-5):
    while abs(b - a) > eps:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def chord(f, a=-10, b=+10, eps=1e-5):
    c = 1.0
    while  abs(f(c)) > eps:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if f(c) * f(b) > 0:
            a = c 
        else: b = c
    return c

def derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x)) / eps

def newton_method(f, x0, eps=1e-5):
    x = x0
    x_old = x - 1
    while abs(x_old - x) > eps:
        x_old = x
        x = x - f(x) / derivative(f, x, eps=eps) 
    return x


a, b = -1, -0.5
bisection_root = bisection(f, a, b)
chord_root = chord(f, b,a)
newton_root = newton_method(f, a)

coeffs = [1, -2, 5, 1, -2, 0]
np_roots = np.roots(coeffs)
np_roots = [root.real for root in np_roots if root.imag == 0]

x_vals = np.linspace(a, b, 1000)
y_vals = f(x_vals)

print(f"""
Range: {a}, {b}
Bisection method root: {bisection_root}
Chord method root:     {chord_root}
Newton method root:    {newton_root}
Numpy roots:           {np_roots}
""")