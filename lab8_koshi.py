import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def f_prime(x: float, y: float) -> float:
    return np.exp(y**2 + b)**(-3.0*x)


# TODO: runge kutta with dynamic step
# TODO: Adams method
def fixed_runge_kutta_4(
    f_prime: Callable[[float, float], float], 
    x0: float,
    y0: float,
    x_end: float, 
    step_size: float
) -> tuple[np.ndarray, np.ndarray]:

    steps = int((x_end - x0) / step_size)
    x = np.linspace(x0, x_end, steps + 1)
    y = np.zeros(steps + 1)
    y[0] = y0

    for i in range(steps):
        k1 = step_size * f_prime(x[i], y[i])
        k2 = step_size * f_prime(x[i] + step_size/2, y[i] + k1/2)
        k3 = step_size * f_prime(x[i] + step_size/2, y[i] + k2/2)
        k4 = step_size * f_prime(x[i] + step_size, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return x, y



def create_phase_portrait(f_prime, y_range, x_range, num_trajectories=20):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    
    Y, X = np.meshgrid(y_vals, x_vals)
    
    U = np.ones_like(X)
    V = f_prime(X, Y)
    
    ax.streamplot(y_vals, x_vals, U, V, density=2, color='gray', linewidth=0.6, arrowsize=0.8)

    y_start = np.linspace(y_range[0], y_range[1], num_trajectories)
    for y0 in y_start:
        x, y = fixed_runge_kutta_4(f_prime, x_range[0], y0, x_range[1], 0.001)
        ax.plot(y, f_prime(x, y), 'b-', linewidth=1, alpha=0.7)
    
    ax.set_xlim(y_range)
    ax.set_ylim(x_range)
    ax.set_xlabel('y')
    ax.set_ylabel('dy/dx')
    ax.set_title('Phase Portrait')
    plt.show()

b = 1.0  
y_range = (-1.2, 1.2)
x_range = (-1.2, 1.2)

create_phase_portrait(f_prime, y_range, x_range)

x0, y0 = 0.0, 1.0
x, y = fixed_runge_kutta_4(f_prime, x0, y0, 1.0, 0.001)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Trajectory')
plt.plot(x[0], y[0], 'ro', label='Start')
plt.title("Single Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()