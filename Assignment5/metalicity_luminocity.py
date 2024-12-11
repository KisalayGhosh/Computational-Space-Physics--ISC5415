import mesa_reader as mr
import numpy as np
import matplotlib.pyplot as plt

# Define the history data files corresponding to each metallicity
history_files = {
    0.01: "history_1M_Z01.data",  # Update path if needed
    0.02: "history_1M_Z02.data",
    0.03: "history_1M_Z03.data"
}

# Initialize lists for metallicities and observed log(L) values
metallicities = []
observed_log_L = []

# Extract log(L) for each metallicity
for metallicity, file_path in history_files.items():
    try:
        # Load history.data file using mesa_reader
        history = mr.MesaData(file_path)
        log_L = history.log_L[-1]  # Get the last logged value of log_L
        metallicities.append(metallicity)
        observed_log_L.append(log_L)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

# Convert observed log(L) to linear scale (L)
observed_L = 10 ** np.array(observed_log_L)

# Compute theoretical luminosity using the provided formula
X = 0.7
Y = 0.28
Z_solar = 0.02
theoretical_L = (np.array(metallicities) / Z_solar) ** (-1 / 4) * (5 * X + 3 * Y) / (2 * X + Y)

# Plot observed vs theoretical luminosity
plt.figure(figsize=(8, 6))
plt.plot(metallicities, observed_L, 'o-', label='Observed Luminosity', markersize=8)
plt.plot(metallicities, theoretical_L, 's-', label='Theoretical Luminosity', markersize=8)
plt.xlabel('Metallicity (Z)')
plt.ylabel('Luminosity (L)')
plt.title('Luminosity-Metallicity Relation for 1 $M_\\odot$ Star')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("luminosity_metallicity_relation_auto.png")  
plt.show()
