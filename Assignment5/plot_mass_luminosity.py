import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt

# List of history files for different masses
history_files = ['history_3M.data', 'history_10M.data', 'history_25M.data']
masses = [3, 10, 25]  # Masses in solar units

log_mass = np.log10(masses)  # Logarithm of masses
log_luminosity = []          # To store log(L) values

# Extract log(L) from each history file
for file in history_files:
    history = mr.MesaData(file)
    log_L = history.log_L[-1]  # Get the last logged value of log_L
    log_luminosity.append(log_L)

# Plot log(M) vs log(L)
plt.figure()
plt.plot(log_mass, log_luminosity, 'o-', label='Mass-Luminosity Relation')
plt.xlabel('log(Mass)')
plt.ylabel('log(Luminosity)')
plt.title('Mass-Luminosity Relation')
plt.legend()
plt.grid(True)
plt.savefig('mass_luminosity_relation.png')  # Save the plot
plt.show()
