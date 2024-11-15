import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Definir parámetros
Lx, Ly = 10, 10  # Tamaño del dominio en 2D
Nx, Ny = 100, 100  # Número de puntos en la malla (más puntos = mayor resolución)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

alpha = 0.01  # Coeficiente de difusión térmica
dt = 0.01  # Paso de tiempo
T_max = 2  # Tiempo máximo de simulación
n_steps = int(T_max / dt)  # Número de pasos de tiempo

# Crear malla de temperatura
T = np.zeros((Nx, Ny))

# Función para pedir al usuario la selección de la condición inicial
def choose_initial_condition():
    print("Seleccione una condición inicial:")
    print("1. Onda Senoidal (Sine)")
    print("2. Distribución Gaussiana (Gaussian)")
    print("3. Personalizar Condición Inicial")
    choice = input("Ingrese el número de su opción (1, 2, o 3): ")

    if choice == '1':
        return 'sine'
    elif choice == '2':
        return 'gaussian'
    elif choice == '3':
        return 'custom'
    else:
        print("Opción no válida. Se seleccionará la onda senoidal por defecto.")
        return 'sine'

# Función para pedir al usuario la selección de las condiciones de contorno
def choose_boundary_condition():
    print("Seleccione el tipo de condición de contorno:")
    print("1. Dirichlet (Temperatura fija en los bordes)")
    print("2. Neumann (Gradiente nulo en los bordes)")
    choice = input("Ingrese el número de su opción (1 o 2): ")

    if choice == '1':
        return 'dirichlet'
    elif choice == '2':
        return 'neumann'
    else:
        print("Opción no válida. Se seleccionará Dirichlet por defecto.")
        return 'dirichlet'

# Función para pedir al usuario una condición inicial personalizada
def get_custom_initial_condition():
    print("Ingrese una expresión matemática para la condición inicial (en función de X y Y):")
    print("Por ejemplo: np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)")
    condition_str = input("Condición inicial personalizada: ")
    return condition_str

# Función para generar la condición inicial basada en el tipo seleccionado
def initial_condition(X, Y, condition_type='sine', t=0, custom_condition=None):
    """
    Devuelve la condición inicial de temperatura.
    
    Parameters:
    X, Y : arrays 2D de las coordenadas espaciales
    condition_type : tipo de condición inicial ('sine', 'gaussian', 'custom')
    t : parámetro de tiempo para desplazamiento de la onda (si aplica)
    custom_condition : string con la fórmula personalizada si se selecciona 'custom'
    
    Returns:
    T : array 2D de temperaturas en las posiciones X, Y
    """
    if condition_type == 'sine':
        return np.sin(np.pi * X / Lx + t * 0.2) * np.sin(np.pi * Y / Ly)
    elif condition_type == 'gaussian':
        return np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.1)
    elif condition_type == 'custom':
        # Evaluar la expresión personalizada del usuario
        return eval(custom_condition)
    else:
        raise ValueError("Tipo de condición inicial no reconocido. Usa 'sine', 'gaussian' o 'custom'.")

# Función para aplicar condiciones de contorno de Dirichlet (temperatura constante en los bordes)
def apply_boundary_conditions(T, boundary_temp=0):
    T[0, :] = boundary_temp  # Borde inferior
    T[-1, :] = boundary_temp  # Borde superior
    T[:, 0] = boundary_temp  # Borde izquierdo
    T[:, -1] = boundary_temp  # Borde derecho
    return T

# Función para aplicar condiciones de contorno de Neumann (gradiente nulo en los bordes)
def apply_neumann_boundary_conditions(T):
    T[0, :] = T[1, :]  # Gradiente nulo en el borde inferior
    T[-1, :] = T[-2, :]  # Gradiente nulo en el borde superior
    T[:, 0] = T[:, 1]  # Gradiente nulo en el borde izquierdo
    T[:, -1] = T[:, -2]  # Gradiente nulo en el borde derecho
    return T

# Función para actualizar la temperatura
def update_temperature(T, alpha, dx, dy, dt):
    T_new = T.copy()
    
    # Ecuación de calor en 2D
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2)
    
    return T_new

# Función para aplicar desplazamiento temporal a la condición inicial
def update_initial_condition(T, X, Y, t, condition_type='sine', custom_condition=None):
    """
    Actualiza la condición inicial con un desplazamiento temporal.
    
    Parameters:
    T : array 2D de la temperatura actual
    X, Y : arrays 2D de las coordenadas espaciales
    t : parámetro de tiempo para el desplazamiento
    condition_type : tipo de condición inicial ('sine', 'gaussian', 'custom')
    custom_condition : fórmula personalizada si la condición es 'custom'
    
    Returns:
    T : array 2D de temperaturas con desplazamiento temporal
    """
    return initial_condition(X, Y, condition_type, t, custom_condition)

# Crear la figura para la animación
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función de animación
def animate(i):
    global T
    
    # Actualizar la temperatura con la ecuación de calor
    T = update_temperature(T, alpha, dx, dy, dt)
    
    # Actualizar la condición inicial con un desplazamiento temporal (dependiendo de la condición)
    T = update_initial_condition(T, X, Y, i, condition_type=initial_type, custom_condition=custom_condition)
    
    # Aplicar condiciones de contorno después de cada actualización
    if boundary_type == 'dirichlet':
        T = apply_boundary_conditions(T, boundary_temp=0)
    elif boundary_type == 'neumann':
        T = apply_neumann_boundary_conditions(T)

    # Limpiar la gráfica anterior
    ax.clear()
    
    # Dibujar la nueva superficie 3D con colormap 'hot' para observar la propagación del calor
    ax.plot_surface(X, Y, T, cmap='hot', edgecolor='none')
    
    # Títulos y etiquetas
    ax.set_title(f'Temperature Distribution at t = {i * dt:.2f} s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')

# Solicitar la elección del usuario para la condición inicial y el contorno
initial_type = choose_initial_condition()

# Si elige 'custom', pedimos al usuario que ingrese una fórmula para la condición inicial
if initial_type == 'custom':
    custom_condition = get_custom_initial_condition()
else:
    custom_condition = None

boundary_type = choose_boundary_condition()

# Crear malla de coordenadas X y Y
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Asignar la condición inicial seleccionada por el usuario
T = initial_condition(X, Y, condition_type=initial_type, t=0, custom_condition=custom_condition)

# Crear la animación
ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=100, repeat=False)

plt.show()
