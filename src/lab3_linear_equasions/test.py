
import numpy as np
matrix_a = np.array([
    [1.787, -0.758, -0.277, 0.127],
    [0.42, 3.95, 1.87, 0.43],
    [0.28, 1.66, 2.31, 0.02],
    [0.88, 0.43, 0.46, 4.44]])



for row in matrix_a:
    print(np.sum(row))