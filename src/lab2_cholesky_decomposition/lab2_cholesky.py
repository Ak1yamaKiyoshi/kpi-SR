import numpy as np
from lib_print import ANSI, printer, highlight, pidx
counter = pidx()

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

print(printer(
    a, default_style=ANSI.Styles.ITALIC,
    text=f"{ANSI.Styles.BOLD}{counter.str()}matrix A{ANSI.Styles.RESET}"), "\n")

print(printer(
    b.reshape(-1, 1), default_style=ANSI.Styles.ITALIC,
    text=f"{ANSI.Styles.BOLD}{counter.str()}vector b{ANSI.Styles.RESET}"), "\n")

h_ij = highlight([], ANSI.FG.GREEN, 0, "case i == j")
h_else = highlight([], ANSI.FG.BLUE, 0, "case else") 
h_ki = highlight([], ANSI.FG.YELLOW, 0, "u[k, i] at else case")
h_kj = highlight([], ANSI.FG.CYAN  , 0, "u[k, j] at else case")

for i in range(n):
    for j in range(i, n):
        if i == j: 
            sum = np.sum([u[k, i]**2 for k in range(i)]) 
            u[i, i] = np.sqrt(a[i, i] - sum)
            h_ij.indicies.append((i, i))
        else:
            sum = np.sum([u[k, i] * u[k, j] for k in range(i)])
            u[i, j] = (a[i, j] - sum) / u[i, i]  # чому тут верхня трикутна матриця 

            for k in range(i):
                h_kj.indicies.append((k, j))
                h_ki.indicies.append((k, i))
            h_else.indicies.append((i, j)) 
print(printer(a, 
        f"{ANSI.Styles.BOLD}{counter.str()}matrix A\n -> (values taken from A at decomposition step) {ANSI.Styles.RESET}", [h_ij, h_else],
        print_description=True, default_style=ANSI.FG.BRIGHT_BLACK), "\n")
print(printer(
        u, formatting="5.5f", higlights=[h_ki], default_style=ANSI.FG.BRIGHT_BLACK, print_description=True,
        text=f"{ANSI.Styles.BOLD}{counter.str()}matrix U (during decomposition){ANSI.Styles.RESET}"), "\n")
print(printer(
        u, formatting="5.5f", higlights=[h_kj], default_style=ANSI.FG.BRIGHT_BLACK, print_description=True,
        text=f"{ANSI.Styles.BOLD}{counter.str()}matrix U (during decomposition){ANSI.Styles.RESET}"), "\n")
print(printer(
        u, formatting="5.5f", default_style=ANSI.Styles.ITALIC,
        text=f"{ANSI.Styles.BOLD}{counter.str()}matrix U{ANSI.Styles.RESET}"), "\n")


# T'y = b
y = np.zeros_like(b)
for i in range(n):
    sum = np.sum([u[k, i] * y[k] for k in range(i)]) 
    y[i] = (b[i] - sum) / u[i, i] 

print(printer(
        y.reshape(-1, 1), default_style=ANSI.FG.GREEN,
        text=f"{ANSI.Styles.BOLD}{counter.str()}vector y{ANSI.Styles.RESET}"), "\n")

# Tx = y    
x = np.zeros_like(b)
for i in range(n-1, -1, -1):
    sum = np.sum([u[i, k] * x[k] for k in range(i+1, n)])       
    x[i] = (y[i] - sum) / u[i, i]

print(printer(
        x.reshape(-1, 1), default_style=ANSI.Styles.ITALIC, formatting="5.5f",
        text=f"{ANSI.Styles.BOLD}{counter.str()}vector x{ANSI.Styles.RESET}"), "\n")
print(printer(
        (a @ x).reshape(-1, 1), default_style=ANSI.Styles.ITALIC, 
        text=f"{ANSI.Styles.BOLD}{counter.str()}a @ x (should be aprox equal b) {ANSI.Styles.RESET}"), "\n")
print(printer(
        (a @ x - b).reshape(-1, 1), default_style=ANSI.Styles.ITALIC, formatting=".6f",
        text=f"{ANSI.Styles.BOLD}{counter.str()}a @ x - b {ANSI.Styles.RESET}"), "\n")
print(printer(
        a - u.T @ u, default_style=ANSI.Styles.ITALIC, formatting=".6f",
        text=f"{ANSI.Styles.BOLD}{counter.str()}a - u.T @ u {ANSI.Styles.RESET}"), "\n")
