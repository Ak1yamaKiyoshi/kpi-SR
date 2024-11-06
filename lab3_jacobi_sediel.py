import numpy as np
from lib_print import * 
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

print(printer(a, f"{ANSI.Styles.BOLD} Matrix A{ANSI.Styles.RESET}", default_style=ANSI.Styles.ITALIC), end="\n\n")
print(printer(b.reshape(-1, 1), f"{ANSI.Styles.BOLD} Vector b{ANSI.Styles.RESET}", default_style=ANSI.Styles.ITALIC), end="\n\n")

def jacobi(a, b, epsilon=1e-5, max_iterations=1000):    
    __print_strouput = f" {ANSI.Styles.BOLD}{ANSI.FG.BLUE}JACOBI ALGORYTHM{ANSI.Styles.RESET}\n\n"
    __default_print_style = ANSI.FG.BRIGHT_BLACK

    n = a.shape[0]
    D_inv = np.diag(1 / np.diag(a))
    iteration_matrix = np.eye(n) - D_inv @ a
    iteration_vector = D_inv @ b
    x = np.zeros_like(b)
    
    # що таке ітераційна матриця та вектор, чому вони використовуються, звідки беруться, чому саме такі.  

    __print_highlights_diag = [highlight([(i, i) for i in range(a.shape[0])], ANSI.FG.BLUE, 1, f"{ANSI.Styles.BOLD}Diagonal ")]
    __print_highlights_not_diag = [highlight([(i, i) for i in range(a.shape[0])], ANSI.FG.BRIGHT_BLACK, 1, f"{ANSI.Styles.BOLD}Diagonal ")]
    __print_strouput += printer(D_inv, ANSI.Styles.BOLD +" (jacobi) D inverted", __print_highlights_diag, default_style=ANSI.Styles.ITALIC+__default_print_style, formatting="+0.5f") + "\n\n"
    __print_strouput += printer(iteration_matrix, ANSI.Styles.BOLD +" (jacobi) iteration matrix ", __print_highlights_not_diag, formatting="+0.5f", default_style=ANSI.Styles.ITALIC) + "\n\n"
    __print_strouput += printer(iteration_vector.reshape(-1, 1), ANSI.Styles.BOLD +" (jacobi) iteration vector ", default_style=ANSI.Styles.ITALIC) + '\n\n'

    for i in range(max_iterations):
        x_new = iteration_matrix @ x + iteration_vector

        error = np.max(np.abs(x_new - x)) 

        __print_strouput += printer(x_new.reshape(-1, 1), f"{ANSI.Styles.BOLD} (jacobi) x at iteration {1+i:2d} with change (x_new - x) {ANSI.FG.RED} {error:0.5f} ", [], "+0.5f", default_style=ANSI.Styles.ITALIC+ANSI.FG.BRIGHT_BLACK, pre_row_str="           ") + '\n\n'

        if error < epsilon:
            __print_strouput += printer((a @ x_new - b).reshape(-1, 1), f"{ANSI.Styles.BOLD} (jacobi) a @ x - b, change (x_new - x): {ANSI.FG.RED} {error:0.5f}{ANSI.Styles.RESET}, {ANSI.FG.BLUE}epsilon: {epsilon:0.5f}", default_style=ANSI.FG.GREEN+ANSI.Styles.ITALIC, formatting="0.5f")
            
            print(__print_strouput)
            
            return x_new
        x = x_new
    
    return x 


def zeidel(a, b, epsilon=1e-5, max_iterations=1000):
    n = a.shape[0]
    x = np.zeros_like(b)
    
    __print_strouput = f" {ANSI.Styles.BOLD}{ANSI.FG.BLUE}Zeidel ALGORYTHM{ANSI.Styles.RESET}\n\n"
    

    for k in range(max_iterations):
        x_new = np.zeros_like(x)
        # що це за внутрішній цикл, навіщо він потрібен 
        for i in range(n):
            sum1 = sum(a[i,j] * x_new[j] for j in range(i)) # що це за сумма, які елементи використовуються та чому 
            sum2 = sum(a[i,j] * x[j] for j in range(i+1, n)) # що це за сумма 
            x_new[i] = (b[i] - sum1 - sum2) / a[i,i]


        error = np.max(np.abs(x_new - x)) # що таке error, та чому воно використовується як умова для зупинення ітерацій
        __print_strouput += printer(x_new.reshape(-1, 1), f"{ANSI.Styles.BOLD} (zeidel) x at iteration {1+k:2d} with change (x_new - x) {ANSI.FG.RED} {error:0.5f} ", [], "+0.5f", default_style=ANSI.FG.BRIGHT_BLACK+ANSI.Styles.ITALIC, pre_row_str="           ") + '\n\n'
        
        if error < epsilon:
            break
        x = x_new

    __print_strouput += printer((a @ x_new - b).reshape(-1, 1), f"{ANSI.Styles.BOLD} (zeidel) a @ x - b, change (x_new - x): {ANSI.FG.RED} {error:0.5f}{ANSI.Styles.RESET}, {ANSI.FG.BLUE}epsilon: {epsilon:0.5f}", default_style=ANSI.Styles.ITALIC+ANSI.FG.GREEN, formatting="0.5f")
    print(__print_strouput)
    return x

jacobi(a, b)
print()
zeidel(a, b)