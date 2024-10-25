import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Constants
q = 0.155  # Mass ratio (Secondary/Primary)
G = 1.0  # Normalized gravitational constant
M1 = 1.0  # Primary mass (normalized)
M2 = q * M1  # Secondary mass
L = 1.0  # Orbital separation (normalized)
softening_length = 1.0e-2  # Softening length to prevent division by zero
min_dist = 1.0e-3  # Minimum distance to avoid overflow

# Initial conditions near Lagrangian point L1
x0 = L * (q / (q + 1))  # near L1
y0 = 0.0
vx0 = 0.0  # initial velocity in x direction
vy0 = np.sqrt(G * M1 / L) * 0.9  # Reduced initial velocity

L1_x = x0  # Lagrangian point L1 in x

# Define the potential function (phi) in a co-rotating frame
def potential(x, y, q):
    r1 = np.sqrt(np.maximum((x + q / (q + 1))**2 + y**2, min_dist**2))  # Distance to M1
    r2 = np.sqrt(np.maximum((x - 1 / (q + 1))**2 + y**2, min_dist**2))  # Distance to M2
    phi = (-G * q / r1 - G / r2 - 0.5 * (x**2 + y**2))  # Effective potential
    return phi

# Derivatives for ODE solver (motion equations in co-rotating frame)
def equations_of_motion(t, w):
    x, y, vx, vy = w
    r1 = np.sqrt(np.maximum((x + q / (q + 1))**2 + y**2, min_dist**2))
    r2 = np.sqrt(np.maximum((x - 1 / (q + 1))**2 + y**2, min_dist**2))

    # Accelerations
    ax = -G * q * (x + q / (q + 1)) / r1**3 - G * (x - 1 / (q + 1)) / r2**3 + x
    ay = -G * q * y / r1**3 - G * y / r2**3 + y

    return [vx, vy, ax, ay]

# Solve the ODEs using an adaptive time-step RK45
def solve_trajectory(t_span, initial_conditions):
    solution = solve_ivp(equations_of_motion, t_span, initial_conditions, method='RK45', rtol=1e-10, atol=1e-12)
    return solution.t, solution.y

# Plotting the Roche Lobe (Potential) and L1 point
def plot_roche_lobe(q):
    x_vals = np.linspace(-1.5, 1.5, 400)
    y_vals = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = potential(X, Y, q)
    
    plt.contourf(X, Y, Z, levels=100, cmap='plasma')
    plt.colorbar(label="Potential")
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Test Particle Orbit and Roche Lobe for q = {q:.3f}")

    # Mark primary, secondary stars and L1 point
    plt.plot(-q / (q + 1), 0, 'bo', label="Primary (M1)")
    plt.plot(1 / (q + 1), 0, 'go', label="Secondary (M2)")
    plt.plot(L1_x, 0, 'ro', label="L1 Point")
    plt.legend()

# Task 3: Calculate Energy and compare orbits around L1 point
def calculate_energy(x, y, vx, vy):
    r1 = np.sqrt((x + q / (q + 1))**2 + y**2)
    r2 = np.sqrt((x - 1 / (q + 1))**2 + y**2)
    kinetic_energy = 0.5 * (vx**2 + vy**2)
    potential_energy = -G * q / r1 - G / r2
    total_energy = kinetic_energy + potential_energy
    return total_energy

def compare_energy(t_vals, solution):
    x_vals = solution[0]  # X positions
    y_vals = solution[1]  # Y positions
    vx_vals = solution[2]  # X velocities
    vy_vals = solution[3]  # Y velocities

    energies = np.array([calculate_energy(x_vals[i], y_vals[i], vx_vals[i], vy_vals[i]) for i in range(len(t_vals))])
    
    plt.figure()
    plt.plot(t_vals, energies, label="Total Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Energy Variation of Particle Near L1 Point")
    plt.legend()
    plt.grid(True)
    plt.show()

# Task 4: Calculate Angular Momentum
def calculate_angular_momentum(x, y, vx, vy):
    Lz = x * vy - y * vx  # z-component of angular momentum
    return Lz

def plot_angular_momentum(t_vals, solution):
    x_vals = solution[0]  # X positions
    y_vals = solution[1]  # Y positions
    vx_vals = solution[2]  # X velocities
    vy_vals = solution[3]  # Y velocities

    angular_momentum = np.array([calculate_angular_momentum(x_vals[i], y_vals[i], vx_vals[i], vy_vals[i]) for i in range(len(t_vals))])

    plt.figure()
    plt.plot(t_vals, angular_momentum, label="Angular Momentum (Lz)", color='blue')
    plt.xlabel("Time")
    plt.ylabel("Angular Momentum")
    plt.title("Angular Momentum Variation of Particle Near L1 Point")
    plt.legend()
    plt.grid(True)
    plt.show()

# Task 5: Calculate Distance from L1
def calculate_distance_from_L1(x, y):
    return np.sqrt((x - L1_x)**2 + y**2)

def plot_distance_from_L1(t_vals, solution):
    x_vals = solution[0]  # X positions
    y_vals = solution[1]  # Y positions

    distances = np.array([calculate_distance_from_L1(x_vals[i], y_vals[i]) for i in range(len(t_vals))])

    plt.figure()
    plt.plot(t_vals, distances, label="Distance from L1")
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.title("Distance of Particle from L1 Point Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Task 6: Velocity Magnitude
def calculate_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)

def plot_velocity_magnitude(t_vals, solution):
    vx_vals = solution[2]  # X velocities
    vy_vals = solution[3]  # Y velocities

    velocities = np.array([calculate_velocity_magnitude(vx_vals[i], vy_vals[i]) for i in range(len(t_vals))])

    plt.figure()
    plt.plot(t_vals, velocities, label="Velocity Magnitude")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.title("Velocity Magnitude of Particle Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# Initial conditions for particle injected near L1 point
initial_conditions = [x0, y0, vx0, vy0]

# Time span for the simulation
t_span = (0, 20)  # Adjust this time span as needed

# Solve the trajectory for particle near L1 point
t_vals, solution = solve_trajectory(t_span, initial_conditions)

x_vals = solution[0]  # X positions
y_vals = solution[1]  # Y positions

# Ensure x_vals and y_vals are sequences (lists or arrays)
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Plotting the Roche Lobe and Particle Trajectory
fig, ax = plt.subplots()
plot_roche_lobe(q)

# Plot particle
particle, = ax.plot([], [], 'ro')

# Update function for particle trajectory
def update(time_idx):
    if time_idx < len(x_vals) and time_idx < len(y_vals):
        particle.set_data([x_vals[time_idx]], [y_vals[time_idx]])  # Wrap as lists
        fig.canvas.draw_idle()

# Add time slider
ax_slider = plt.axes([0.2, 0.01, 0.65, 0.03])
slider_time = Slider(ax_slider, 'Time', 0, len(t_vals) - 1, valinit=0, valstep=1)
slider_time.on_changed(lambda val: update(int(val)))

# Initialize plot with the first frame
update(0)

# Display trajectory and Roche lobe plot
plt.show()

# Additional Plots
compare_energy(t_vals, solution)
plot_angular_momentum(t_vals, solution)
plot_distance_from_L1(t_vals, solution)
plot_velocity_magnitude(t_vals, solution)
