import numpy as np
from datetime import datetime, timedelta

def parse_ephemeris(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the $$SOE (start of ephemeris data) and $$EOE (end of ephemeris data)
    start_idx = lines.index("$$SOE\n") + 1
    end_idx = lines.index("$$EOE\n")

    positions = []
    velocities = []
    times = []

    for line in lines[start_idx:end_idx]:
        parts = line.split()
        jd_time = float(parts[0])  # Julian Date Time
        vx, vy, vz = map(float, parts[1:4])  # Velocity components in km/s

        # Convert velocity from km/s to cm/s
        velocities.append([vx * 1e5, vy * 1e5, vz * 1e5])

        # Approximate position using a fixed step in time (1 day, 86400 seconds)
        # Note: We will calculate positions using the initial conditions in the simulation
        times.append(jd_time)

    return np.array(velocities), times

# Example usage
velocities_earth, times_earth = parse_ephemeris('horizons_results_earth.txt')
velocities_mercury, times_mercury = parse_ephemeris('horizons_results_mercury.txt')
velocities_jupiter, times_jupiter = parse_ephemeris('horizons_results_jupiter.txt')
velocities_neptune, times_neptune = parse_ephemeris('horizons_results_neptune.txt')
velocities_moon, times_moon = parse_ephemeris('horizons_results_moon.txt')
velocities_sun, times_sun = parse_ephemeris('horizons_results_sun.txt')
