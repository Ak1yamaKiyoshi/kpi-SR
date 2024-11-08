import numpy as np
from typing import Callable
import warnings
warnings.filterwarnings("ignore")


def f2(x, y: np.ndarray) -> np.ndarray:
    y0, y1 = y
    #return np.array([np.cos(2*y0), np.exp(np.sin(x))**y1])
    return np.array([np.cos(np.log1p(y1**2)), np.exp(3)**np.cos(y0)])


def plot_fancy_phase_portrait(f2):
    import matplotlib.pyplot as plt
    x = np.array([0.1, 0.0]) 
    x_vals = np.linspace(-3.0, 3.0, 30)
    y_vals = np.linspace(-3.0, 3.0, 30)
    X, Y = np.meshgrid(x_vals, y_vals)
    u, v = np.zeros_like(X), np.zeros_like(X)
    NI, NJ = X.shape

    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            fp = f2(0, [x, y])
            u[i, j] = fp[0]  # dx/dt
            v[i, j] = fp[1]  # dy/dt

    speed = np.sqrt(u**2 + v**2)  # Magnitude of the velocity at each point
    plt.streamplot(X, Y, u, v, color=speed, linewidth=1, cmap='coolwarm')
    plt.axis('square')
    plt.axis([-3, 3, -3, 3])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    plt.tight_layout()

plot_fancy_phase_portrait(f2)