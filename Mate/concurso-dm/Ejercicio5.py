import numpy as np
import matplotlib.pyplot as plt

# Ecuación diferencial: x'(t) = 2*x(t)+1
def f(t, x):
    return 2*x+1

# df/dt = Grad(f)*(1, x') = (0, 2)*(1, f) = 2*f
def f_prima(t, x):
    return 2*f(t, x)

# Solución exacta de la ecuación diferencial
def solucion_exacta(t, x0):
    return (x0+0.5)*np.exp(2*t) - 0.5

# Función para el método de Euler de orden 2
def euler_method_order_2(f, t0, x0, tF, h):
    t_values = [t0]
    x_values = [x0]
    
    t = t0
    x = x0
    
    while t < tF:
        x += (h/2) * (f(t, x) + f(t + h, x + h * f(t, x)))
        t += h
        t_values.append(t)
        x_values.append(x)
    
    return t_values, x_values

# Función para el método de Taylor de orden 2
def taylor_method(f, f_prima, t0, x0, tF, h):
    t_values = [t0]
    x_values = [x0]
    
    t = t0
    x = x0
    
    while t < tF:
        x += h * f(t, x) + (h**2 / 2) * (f_prima(t, x))
        t += h
        t_values.append(t)
        x_values.append(x)
    
    return t_values, x_values

h_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

errors_euler = []
errors_taylor = []

for h in h_values:
    t_f = 1.0
    t_values_euler, x_values_euler = euler_method_order_2(f, 0, 1, t_f, h)
    t_values_taylor, x_values_taylor = taylor_method(f, f_prima, 0, 1, t_f, h)
    plt.title(f"h={h}")
    plt.plot(t_values_euler, x_values_euler, label="Euler")
    plt.plot(t_values_euler, x_values_taylor, label="Taylor")
    plt.plot(t_values_euler, [solucion_exacta(t, 1) for t in t_values_euler], label="Exacta")
    plt.legend()
    plt.show()
    plt.close()
    
    error_euler = abs(solucion_exacta(t_f, 1) - x_values_euler[-1])
    error_taylor = abs(solucion_exacta(t_f, 1) - x_values_taylor[-1])
    
    errors_euler.append(error_euler)
    errors_taylor.append(error_taylor)

# Graficar log(EN) en función de log(h)
plt.scatter(-np.log(h_values), np.log(errors_euler), label='Error Euler')
plt.scatter(-np.log(h_values), np.log(errors_taylor), label='Error Taylor')
plt.xlabel('-log(h)')
plt.ylabel('log(EN)')
plt.legend()
plt.grid(True)
plt.show()
