import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the differential equation dy/dt = -2y
def dydt(t, y):
    return -2 * y

# Initial condition
y0 = [1]

# Define the time span for the solution (from t=0 to t=5)
t_span = (0, 5)

# Solve the Cauchy problem
solution = solve_ivp(dydt, t_span, y0, method='RK45', t_eval=np.linspace(0, 5, 100))

# Extract time and solution
t = solution.t
y = solution.y[0]

# Plotting the solution
plt.plot(t, y, label=r"$\frac{dy}{dt} = -2y, y(0)=1$")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title("Solution of the Cauchy Problem")
plt.grid()
plt.legend()
plt.show()
