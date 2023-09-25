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
    
    
def euler_method(f, t0, x0, tF, h):
    t_values = [t0]
    x_values = [x0]
    
    t = t0
    x = x0
    
    while t < tF:
        x += h * f(t, x)
        t += h
        
        t_values.append(t)
        x_values.append(x)
        
    return t_values, x_values
    
t_values_euler, x_values_euler = euler_method(f, 0, 1, 1, h=0.01)

plt.plot(t_values_euler, x_values_euler, label="Taylor")
plt.plot(t_values_euler, [solucion_exacta(t, 1) for t in t_values_euler], label="Exacta")
plt.legend()
plt.show()
plt.close()
  
