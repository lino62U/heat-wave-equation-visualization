import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Parámetros generales
Lx, Ly = 10, 10  # Tamaño del dominio en 2D
Nx, Ny = 100, 100  # Número de puntos en la malla
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

c = 1.0  # Velocidad de propagación
dt = 0.01  # Paso de tiempo
r = (c * dt) / dx  # Número de Courant

# Parámetros de la simulación
T_max = 2  # Tiempo máximo de simulación
n_steps = int(T_max / dt)  # Número de pasos de tiempo

# Crear la malla de temperaturas u[i,j]
u = np.zeros((Nx, Ny))
u_old = np.zeros((Nx, Ny))  # Almacenará los valores u[i,j-1]
u_new = np.zeros((Nx, Ny))  # Almacenará los valores u[i,j+1]

# Función para pedir al usuario la selección de la condición inicial
def choose_initial_condition():
    print("Seleccione una condición inicial:")
    print("1. Onda Senoidal (Sine)")
    print("2. Distribución Gaussiana (Gaussian)")
    choice = input("Ingrese el número de su opción (1 o 2): ")

    if choice == '1':
        return 'sine'
    elif choice == '2':
        return 'gaussian'
    else:
        print("Opción no válida. Se seleccionará la onda senoidal por defecto.")
        return 'sine'

# Función para aplicar las condiciones de contorno
def apply_boundary_conditions(u):
    u[0, :] = 0  # Borde inferior
    u[-1, :] = 0  # Borde superior
    u[:, 0] = 0  # Borde izquierdo
    u[:, -1] = 0  # Borde derecho
    return u

# Función para actualizar la solución de la ecuación de onda
def update_wave(u, u_old, u_new, r, Nx, Ny):
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u_new[i, j] = 2 * (1 - r**2) * u[i, j] + r**2 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]) - u_old[i, j]
    return u_new

# Función para generar la condición inicial (onda seno)
def initial_condition(X, Y, t=0, condition_type='sine'):
    if condition_type == 'sine':
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.cos(2 * np.pi * t)  # Añadir coseno para el movimiento
    elif condition_type == 'gaussian':
        return np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.1)

# Crear malla de coordenadas X y Y
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Selección de la condición inicial
initial_type = choose_initial_condition()

# Asignar la condición inicial
u = initial_condition(X, Y, t=0, condition_type=initial_type)
u_old = u.copy()  # Inicializamos u_old
u_new = np.copy(u)  # Inicializamos u_new

# Crear la figura para la animación en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función de animación
def animate(i):
    global u, u_old, u_new
    
    # Actualizar la onda
    u_new = update_wave(u, u_old, u_new, r, Nx, Ny)
    
    # Aplicar condiciones de contorno
    u_new = apply_boundary_conditions(u_new)
    
    # Actualizar los valores de u_old y u
    u_old = u.copy()
    u = u_new.copy()
    
    # Limpiar el gráfico y actualizar la superficie 3D
    ax.clear()

    # Crear la superficie 3D con la onda actualizada
    surf = ax.plot_surface(X, Y, u, cmap='viridis', edgecolor='none')

    # Títulos y etiquetas
    ax.set_title(f'Onda en t = {i * dt:.2f} s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Amplitud')

    # Devolver la superficie para la animación
    return [surf]

# Crear la animación
ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100, blit=False)

plt.show()
