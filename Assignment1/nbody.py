import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
import time

# Vector class for 3D operations
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, v):
        return Vec3(self.x + v.x, self.y + v.y, self.z + v.z)

    def __sub__(self, v):
        return Vec3(self.x - v.x, self.y - v.y, self.z - v.z)

    def __mul__(self, n):
        return Vec3(self.x * n, self.y * n, self.z * n)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def get_length(self):
        return np.sqrt(self.dot(self))

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_array(arr):
        return Vec3(arr[0], arr[1], arr[2])

# Particle class for 3D
class Particle:
    def __init__(self, initial_pos, initial_vel, mass):
        self.pos = initial_pos
        self.vel = initial_vel
        self.mass = mass

# Gravitational force calculation with softening length
def calculate_accelerations(particles, softening_length, G=1):
    n = len(particles)
    acc = np.zeros((n, 3))  # 3D acceleration

    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = particles[j].pos.to_array() - particles[i].pos.to_array()
                r = np.linalg.norm(r_vec)
                # Add softening length to avoid singularity when particles are too close
                softened_r = np.sqrt(r**2 + softening_length**2)
                acc[i] += G * particles[j].mass * r_vec / softened_r**3

    return acc

# ODE system of equations for 3D
def vectorfield(t, var, particles, softening_length, G):
    n = len(particles)

    # Reshape input vector to positions and velocities
    pos = var[:3 * n].reshape(n, 3)
    vel = var[3 * n:].reshape(n, 3)

    # Update positions in particle objects
    for i in range(n):
        particles[i].pos = Vec3.from_array(pos[i])

    # Calculate accelerations
    acc = calculate_accelerations(particles, softening_length, G)

    # Flatten positions, velocities, and accelerations for the ODE solver
    dpos_dt = vel.flatten()
    dvel_dt = acc.flatten()

    return np.concatenate([dpos_dt, dvel_dt])

# Main simulation function with energy and angular momentum tracking
def run_simulation(particles, t_end=60.0, steps=800, softening_length=0.02, G=1, reverse=False, rtol=1e-10, atol=1e-10):
    n = len(particles)

    # Initial conditions
    var = np.zeros(6 * n)  # 3 positions + 3 velocities for each particle
    for i, p in enumerate(particles):
        var[3 * i:3 * i + 3] = p.pos.to_array()
        var[3 * n + 3 * i:3 * n + 3 * i + 3] = p.vel.to_array()

    # Time array
    t_span = (0, t_end) if not reverse else (t_end, 0)
    t_eval = np.linspace(t_span[0], t_span[1], steps + 1)

    # Record start time for wall clock measurement
    start_time = time.time()

    # Solve the system of ODEs using RK45 with specified error tolerance
    sol = solve_ivp(vectorfield, t_span, var, args=(particles, softening_length, G), method='RK45',
                    t_eval=t_eval, rtol=rtol, atol=atol)

    # Extract positions and velocities from the solution
    positions = sol.y[:3 * n].reshape(n, 3, steps + 1)
    velocities = sol.y[3 * n:].reshape(n, 3, steps + 1)

    # Track energy and angular momentum over time
    energy_over_time = []
    angular_momentum_over_time = []

    for i in range(steps + 1):
        energy = calculate_total_energy(particles, positions[:, :, i], velocities[:, :, i], G)
        angular_momentum = calculate_angular_momentum(particles, positions[:, :, i], velocities[:, :, i])
        energy_over_time.append(energy)
        angular_momentum_over_time.append(angular_momentum)

    # Record end time for wall clock measurement
    end_time = time.time()
    total_time = end_time - start_time

    return sol.t, positions, velocities, total_time, sol.nfev, energy_over_time, angular_momentum_over_time

# Energy calculation (total energy = kinetic + potential)
def calculate_total_energy(particles, positions, velocities, G=1):
    n = len(particles)
    kinetic_energy = 0
    potential_energy = 0

    for i in range(n):
        vel = np.linalg.norm(velocities[i])
        kinetic_energy += 0.5 * particles[i].mass * vel**2

        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            potential_energy -= G * particles[i].mass * particles[j].mass / r

    return kinetic_energy + potential_energy

# Angular momentum calculation
def calculate_angular_momentum(particles, positions, velocities):
    n = len(particles)
    total_angular_momentum = np.zeros(3)

    for i in range(n):
        r = positions[i]
        v = velocities[i]
        m = particles[i].mass
        total_angular_momentum += m * np.cross(r, v)

    return np.linalg.norm(total_angular_momentum)

# Visualization function for 3D particle trajectories
def visualize_simulation(t, positions, particles, t_end):
    n = len(particles)

    # Setup the plot (for 2D projections of 3D motion)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    ax.set_title("N-body Simulation (Projection)")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Circles and lines to represent particles and their trajectories
    circles = [plt.Circle((positions[i, 0, 0], positions[i, 1, 0]), 0.2, ec="w", lw=2.5, zorder=20) for i in range(n)]
    lines = [ax.plot(positions[i, 0, :1], positions[i, 1, :1])[0] for i in range(n)]

    for circle in circles:
        ax.add_patch(circle)

    # Slider to control the time
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
    slider = Slider(slider_ax, 'Time', 0, t_end, valinit=0)

    # Update function for the slider
    def update(time):
        max_index = len(t) - 1  # Ensure the maximum index is within bounds
        i = int(np.rint(time * max_index / t_end))  # Time step index capped by max_index

        if i >= max_index:
            i = max_index

        # Adjust the axis limits dynamically based on current positions
        all_x = np.concatenate([positions[j, 0, :i + 1] for j in range(n)])
        all_y = np.concatenate([positions[j, 1, :i + 1] for j in range(n)])

        ax.set_xlim(all_x.min() - 1, all_x.max() + 1)  # Expand with some margin
        ax.set_ylim(all_y.min() - 1, all_y.max() + 1)  # Expand with some margin

        for j in range(n):
            circles[j].center = positions[j, 0, i], positions[j, 1, i]
            lines[j].set_xdata(positions[j, 0, :i + 1])
            lines[j].set_ydata(positions[j, 1, :i + 1])

    slider.on_changed(update)
    plt.show()

# Function to plot energy and angular momentum over time
def plot_energy_and_angular_momentum(time, energy_over_time, angular_momentum_over_time):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot energy
    ax1.plot(time, energy_over_time, label='Total Energy', color='b')
    ax1.set_title('Total Energy Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy')
    ax1.grid(True)
    ax1.legend()

    # Plot angular momentum
    ax2.plot(time, angular_momentum_over_time, label='Total Angular Momentum', color='r')
    ax2.set_title('Total Angular Momentum Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Angular Momentum')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Input function to read initial conditions from a text file
def read_input_file(filename):
    particles = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        softening_length = float(lines[0].strip())
        for line in lines[1:]:
            data = list(map(float, line.strip().split()))
            pos = Vec3(data[0], data[1], data[2])
            vel = Vec3(data[3], data[4], data[5])
            mass = data[6]
            particles.append(Particle(pos, vel, mass))
    return particles, softening_length

# Main code execution
if __name__ == "__main__":
    # Read particles and softening length from input file
    particles, softening_length = read_input_file('input.txt')

    t_end = 70.0 
    # Run the simulation (forward)
    t, positions, velocities, total_time, nfev, energy_over_time, angular_momentum_over_time = run_simulation(
        particles, t_end=t_end, softening_length=softening_length
    )

    # Visualize the results
    visualize_simulation(t, positions, particles, t_end=t_end)

    # Plot energy and angular momentum over time
    plot_energy_and_angular_momentum(t, energy_over_time, angular_momentum_over_time)

    # Calculate and print final energy and angular momentum
    print(f"Total Energy at Final Step: {energy_over_time[-1]}")
    print(f"Total Angular Momentum at Final Step: {angular_momentum_over_time[-1]}")
    print(f"Total Time for Simulation: {total_time} seconds")
    print(f"Total Function Evaluations: {nfev}")

    # Backward simulation (starting from final state)
    t_back, positions_back, velocities_back, total_time_back, nfev_back, _, _ = run_simulation(
        particles, t_end=t_end, softening_length=softening_length, reverse=True
    )

    # Compare initial conditions recovery
    recovered_initial_positions = positions_back[:, :, -1]
    recovered_initial_velocities = velocities_back[:, :, -1]
    original_positions = np.array([p.pos.to_array() for p in particles])
    original_velocities = np.array([p.vel.to_array() for p in particles])

    position_error = np.linalg.norm(original_positions - recovered_initial_positions)
    velocity_error = np.linalg.norm(original_velocities - recovered_initial_velocities)

    print(f"Position Error After Backward Integration: {position_error}")
    print(f"Velocity Error After Backward Integration: {velocity_error}")
    print(f"Total Time for Backward Simulation: {total_time_back} seconds")
    print(f"Total Function Evaluations (Backward): {nfev_back}")
