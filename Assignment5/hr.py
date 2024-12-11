import numpy as np
import matplotlib.pyplot as plt

# Define metallicities (Z values)
metallicities = np.array([0.01, 0.02, 0.03])  # Z values
X = 0.7  # Hydrogen fraction
Y_values = 1 - X - metallicities  # Compute helium fractions

# Theoretical Luminosity Calculation
theoretical_L = (
    X ** (-2 / 13) *
    (1 + X) ** (-14 / 13) *
    (5 * X + 3) ** (-101 / 13) *
    metallicities ** (-14 / 13)
)

# Placeholder for observed log(L) values (replace with actual data extracted from MESA)
observed_log_L = np.array([0.3, 0.2, 0.1])  # Replace with actual values
observed_L = 10 ** observed_log_L  # Convert log(L) to linear scale

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(metallicities, theoretical_L, 's-', label='Theoretical Luminosity', markersize=8)
plt.plot(metallicities, observed_L, 'o-', label='Observed Luminosity', markersize=8)
plt.xlabel('Metallicity (Z)')
plt.ylabel('Luminosity (L)')
plt.title('Luminosity-Metallicity Relation for 1 $M_\\odot$ Star')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = "luminosity_metallicity_relation.png"
plt.savefig(output_path)
plt.show()

output_path
