import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

# Vector class for 3D operations
class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Overloaded addition operator for vector addition
    def __add__(self, v):
        return Vec3(self.x + v.x, self.y + v.y, self.z + v.z)

    # Overloaded subtraction operator for vector subtraction
    def __sub__(self, v):
        return Vec3(self.x - v.x, self.y - v.y, self.z - v.z)

    # Overloaded multiplication operator for scalar multiplication
    def __mul__(self, n):
        return Vec3(self.x * n, self.y * n, self.z * n)

    # Method to calculate the dot product of two vectors
    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    # Method to get the magnitude of the vector
    def get_length(self):
        return np.sqrt(self.dot(self))

    # Method to convert Vec3 object to a NumPy array
    def to_array(self):
        return np.array([self.x, self.y, self.z])

    # Static method to create a Vec3 object from a NumPy array
    @staticmethod
    def from_array(arr):
        return Vec3(arr[0], arr[1], arr[2])

# Particle class for representing celestial bodies
class Particle:
    def __init__(self, initial_pos, initial_vel, mass):
        self.pos = initial_pos  # Position as a Vec3 object
        self.vel = initial_vel  # Velocity as a Vec3 object
        self.mass = mass        # Mass of the particle

# Function to calculate gravitational accelerations
def calculate_accelerations(particles, softening_length, G):
    n = len(particles)  # Number of particles
    acc = np.zeros((n, 3))  # Initialize acceleration array

    # Loop through all particle pairs to calculate acceleration
    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid self-interaction
                r_vec = particles[j].pos.to_array() - particles[i].pos.to_array()
                r = np.linalg.norm(r_vec)
                softened_r = np.sqrt(r**2 + softening_length**2)  # Softened distance
                acc[i] += G * particles[j].mass * r_vec / softened_r**3  # Gravitational force

    return acc

# Function to define the ODE system of equations
def vectorfield(t, var, particles, softening_length, G):
    n = len(particles)  # Number of particles
    pos = var[:3 * n].reshape(n, 3)  # Extract positions from state vector
    vel = var[3 * n:].reshape(n, 3)  # Extract velocities from state vector

    # Update particle positions for acceleration calculation
    for i in range(n):
        particles[i].pos = Vec3.from_array(pos[i])

    acc = calculate_accelerations(particles, softening_length, G)  # Compute accelerations
    return np.concatenate([vel.flatten(), acc.flatten()])  # Return derivatives as a flattened array

# Main function to run the simulation using solve_ivp
def run_simulation(particles, t_end=365.25, steps=800, softening_length=1e-4, G=6.67430e-8, rtol=1e-10, atol=1e-10):
    n = len(particles)  # Number of particles
    var = np.zeros(6 * n)  # State vector to store positions and velocities

    # Initialize state vector with particle positions and velocities
    for i, p in enumerate(particles):
        var[3 * i:3 * i + 3] = p.pos.to_array()
        var[3 * n + 3 * i:3 * n + 3 * i + 3] = p.vel.to_array()

    t_span = (0, t_end * 24 * 3600)  # Time span in seconds
    t_eval = np.linspace(t_span[0], t_span[1], steps + 1)  # Time evaluation points

    # Solve the ODE using RK45 method
    start_time = time.time()
    sol = solve_ivp(vectorfield, t_span, var, args=(particles, softening_length, G),
                    method='RK45', t_eval=t_eval, rtol=rtol, atol=atol)
    end_time = time.time()

    # Reshape the solution arrays to extract positions and velocities
    positions = sol.y[:3 * n].reshape(n, 3, steps + 1)
    velocities = sol.y[3 * n:].reshape(n, 3, steps + 1)
    return sol.t, positions, velocities, end_time - start_time, sol.nfev

# Function to calculate total energy of the system
def calculate_total_energy(particles, positions, velocities, G):
    n = len(particles)
    kinetic_energy = 0
    potential_energy = 0

    # Calculate kinetic energy
    for i in range(n):
        v = np.linalg.norm(velocities[i])
        kinetic_energy += 0.5 * particles[i].mass * v**2

        # Calculate potential energy for each unique particle pair
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            potential_energy -= G * particles[i].mass * particles[j].mass / r

    return kinetic_energy + potential_energy

# Function to calculate total angular momentum of the system
def calculate_angular_momentum(particles, positions, velocities):
    n = len(particles)
    angular_momentum = np.zeros(3)

    # Calculate angular momentum for each particle
    for i in range(n):
        r = positions[i]
        v = velocities[i]
        m = particles[i].mass
        angular_momentum += m * np.cross(r, v)

    return np.linalg.norm(angular_momentum)

# Function to plot orbits with slider and zoom functionality
def plot_orbits_with_slider_and_zoom(t, positions, labels):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.set_title('Planetary Orbits')
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')

    # Set axis limits based on the maximum range of positions
    max_range = np.max(np.abs(positions))
    ax.set_xlim(-1.1 * max_range, 1.1 * max_range)
    ax.set_ylim(-1.1 * max_range, 1.1 * max_range)
    ax.axis('equal')

    # Initialize plot lines for each celestial body
    lines = [ax.plot([], [], 'o', label=label)[0] for label in labels]
    ax.legend()

    # Function to update plot based on slider value
    def update_plot(step):
        for i, line in enumerate(lines):
            line.set_data([positions[i, 0, int(step)]], [positions[i, 1, int(step)]])
        fig.canvas.draw_idle()

    # Slider widget for time control
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time', 0, len(t) - 1, valinit=0, valstep=1)
    slider.on_changed(update_plot)

    # Function to handle zooming with the mouse scroll wheel
    def zoom(event):
        base_scale = 1.1
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        # Determine zoom direction
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Calculate new axis limits based on zoom
        new_xlim = [xdata - (xdata - cur_xlim[0]) * scale_factor,
                    xdata + (cur_xlim[1] - xdata) * scale_factor]
        new_ylim = [ydata - (ydata - cur_ylim[0]) * scale_factor,
                    ydata + (cur_ylim[1] - ydata) * scale_factor]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.figure.canvas.draw()

    # Connect zoom function to scroll event
    fig.canvas.mpl_connect('scroll_event', zoom)

    update_plot(0)
    plt.show()

# Plotting functions for energy and angular momentum
def plot_energy_and_angular_momentum(t, energy_over_time, angular_momentum_over_time):
    plt.figure()
    plt.plot(t, energy_over_time, label='Total Energy')
    plt.plot(t, angular_momentum_over_time, label='Angular Momentum')
    plt.title('Energy and Angular Momentum Conservation')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for relative distance error
def plot_relative_distance_error(t, positions, initial_distances, labels):
    plt.figure()
    for i, label in enumerate(labels[1:]):  # Exclude the Sun
        current_distances = np.linalg.norm(positions[i + 1, :, :], axis=0)  # Calculate distance over time
        relative_error = np.abs((current_distances - initial_distances[i]) / initial_distances[i])
        plt.plot(t, relative_error, label=f'Relative Distance Error: {label}')
    plt.title('Relative Distance Error from Sun')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function for total energy relative error
def plot_total_energy_error(t, energy_over_time):
    initial_energy = energy_over_time[0]
    relative_error = np.abs((energy_over_time - initial_energy) / initial_energy)
    plt.figure()
    plt.plot(t, relative_error, label='Relative Total Energy Error')
    plt.title('Total Energy Relative Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to initialize celestial bodies with mass, position, and velocity we got  from the horizons data 
def initialize_particles():
    # Sun
    sun_mass = 1.989e33
    sun_pos = Vec3(0, 0, 0)
    sun_vel = Vec3(0, 0, 0)

    # Mercury
    mercury_mass = 3.285e26
    mercury_pos = Vec3(5.79e12, 0, 0)
    mercury_vel = Vec3(0, 4.79e6, 0)

    # Earth
    earth_mass = 5.972e27
    earth_pos = Vec3(1.496e13, 0, 0)
    earth_vel = Vec3(0, 2.978e6, 0)

    # Moon
    moon_mass = 7.348e25
    moon_pos = Vec3(1.496e13 + 3.844e10, 0, 0)
    moon_vel = Vec3(0, 2.978e6 + 1.022e5, 0)

    # Jupiter
    jupiter_mass = 1.898e30
    jupiter_pos = Vec3(7.785e13, 0, 0)
    jupiter_vel = Vec3(0, 1.307e6, 0)

    # Neptune
    neptune_mass = 1.024e30
    neptune_pos = Vec3(4.495e14, 0, 0)
    neptune_vel = Vec3(0, 5.43e5, 0)

    # List of all particles
    particles = [
        Particle(sun_pos, sun_vel, sun_mass),
        Particle(mercury_pos, mercury_vel, mercury_mass),
        Particle(earth_pos, earth_vel, earth_mass),
        Particle(moon_pos, moon_vel, moon_mass),
        Particle(jupiter_pos, jupiter_vel, jupiter_mass),
        Particle(neptune_pos, neptune_vel, neptune_mass)
    ]

    return particles

# Main execution
if __name__ == "__main__":
    particles = initialize_particles()  # Initialize particles
    t, positions, velocities, total_time, nfev = run_simulation(particles)  # Run simulation

    # Calculate energy and angular momentum over time
    energy_over_time = [calculate_total_energy(particles, positions[:, :, i], velocities[:, :, i], 6.67430e-8) for i in range(len(t))]
    angular_momentum_over_time = [calculate_angular_momentum(particles, positions[:, :, i], velocities[:, :, i]) for i in range(len(t))]
    initial_distances = [np.linalg.norm(particles[i + 1].pos.to_array()) for i in range(len(particles) - 1)]  # Initial distances from the Sun

    labels = ['Sun', 'Mercury', 'Earth', 'Moon', 'Jupiter', 'Neptune']
    plot_orbits_with_slider_and_zoom(t, positions, labels)  # Plot orbits with slider and zoom
    plot_energy_and_angular_momentum(t, energy_over_time, angular_momentum_over_time)  # Plot energy and angular momentum
    plot_relative_distance_error(t, positions, initial_distances, labels)  # Plot relative distance error
    plot_total_energy_error(t, energy_over_time)  # Plot total energy relative error

    # Print total simulation time and number of function evaluations
    print(f"Total Time for Simulation: {total_time} seconds")
    print(f"Function Evaluations: {nfev}")
