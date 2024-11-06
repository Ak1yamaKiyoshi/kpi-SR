import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
import warnings
warnings.filterwarnings("ignore")

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
        x, y = runga(f_prime, x_range[0], y0, x_range[1], 0.001)
        ax.plot(y, f_prime(x, y), 'b-', linewidth=1, alpha=0.7)

    ax.set_xlim(y_range)
    ax.set_ylim(x_range)
    ax.set_xlabel('y')
    ax.set_ylabel('dy/dx')
    ax.set_title('Phase Portrait')
    plt.show()


def runga(f, t0, x0, dt, total_steps, tolerance=1e-5 ):
    t = t0
    x = x0
    t_last = x0 + total_steps * dt

    t_hist = []
    x_hist = []
    
    tau_upper = 1e-1
    tau_lower = 4e-3

    while t < t_last:
        k1 = dt * f(t,          x)

        # current step size s
        k2 = dt * f(t + dt / 2, x + k1 / 2)
        k3 = dt * f(t + dt / 2, x + k2 / 2)
        k4 = dt * f(t + dt,     x + k3)
        cur_step_size_x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # double smaller 
        k2 = dt * f(t + dt / 4, x + k1 / 4)
        k3 = dt * f(t + dt / 4, x + k2 / 4)
        k4 = dt * f(t + dt / 2,     x + k3 / 2)
        smaller_step_size = x + (k1 + 2 * k2 + 2 * k3 + k4) / 12

        # double bigger 
        k2 = dt * f(t + dt,     x + k1)
        k3 = dt * f(t + dt,     x + k2)
        k4 = dt * f(t + dt * 2, x + k3 * 2)
        bigger_step_size = x + (k1 + 2 * k2 + 2 * k3 + k4) / 3
        
        x_new = cur_step_size_x
        tau = np.max(np.abs(k2 - k3) / (k1 -  k2 + 1e-15))

        if tau < tau_lower:
            dt *= 2
            x_new = bigger_step_size
 
        elif tau < tau_upper:
            dt /= 2
            x_new = smaller_step_size

        x = x_new 
        t += dt
        t_hist.append(t)
        x_hist.append(x)
    return np.array(t_hist), np.array(x_hist)


def adams_method(f, x0, y0, dt, n):
    h = dt
    xf = n * dt + x0
    n = int(n)
    x = np.linspace(x0, xf, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(3):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    for i in range(3, n):
        y_pred = y[i] + h/24 * (55*f(x[i], y[i]) - 59*f(x[i-1], y[i-1]) + 
                                37*f(x[i-2], y[i-2]) - 9*f(x[i-3], y[i-3]))

        y[i+1] = y[i] + h/24 * (9*f(x[i+1], y_pred) + 19*f(x[i], y[i]) - 
                                5*f(x[i-1], y[i-1]) + f(x[i-2], y[i-2]))

    return x, y

b = 1.0  
y_range = (-1.2, 1.2)
x_range = (-1.2, 1.2)


create_phase_portrait(f_prime, y_range, x_range)


stepsize = 0.001
x0, y0 = 0.0, 1
x, y = adams_method(f_prime, x0, y0, stepsize, b//stepsize)

plt.figure(figsize=(10, 6))
plt.plot(x, y, '.', marker="x", markersize=4)
plt.plot(x, y, '-', alpha=0.5)
plt.plot(x[0], y[0], 'ro')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()