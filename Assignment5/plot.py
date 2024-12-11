import numpy as np
import matplotlib.pyplot as plt
import mesa_reader as mr

# Define stellar masses and their corresponding history files
stellar_masses = np.array([3, 10, 25])  # Stellar masses in solar masses
history_files = {
    3: "history_3M.data",
    10: "history_10M.data",
    25: "history_25M.data"
}

# Theoretical lifetime calculation function
def compute_t_MS(M):
    numerator = 2.5e3 + 6.7e2 * M**2.5 + M**4.5
    denominator = 3.3e-2 * M**1.5 + 3.5e-1 * M**4.5
    return numerator / denominator

# Compute theoretical lifetimes
theoretical_lifetimes = compute_t_MS(stellar_masses)

# Extract observed lifetimes from MESA history files
observed_lifetimes = []
for mass, file_path in history_files.items():
    try:
        # Load history.data file using mesa_reader
        history = mr.MesaData(file_path)
        t_MS = history.star_age[-1]  # Get the final age (last step) as t_MS
        observed_lifetimes.append(t_MS)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        observed_lifetimes.append(np.nan)  # Use NaN for missing data

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(stellar_masses, theoretical_lifetimes, 'o-', label='Theoretical $t_{MS}$', markersize=8)
plt.plot(stellar_masses, observed_lifetimes, 's-', label='Observed $t_{MS}$', markersize=8)
plt.xlabel('Stellar Mass ($M/M_\\odot$)')
plt.ylabel('Main Sequence Lifetime ($t_{MS}$ in Myr)')
plt.title('Main Sequence Lifetime for Solar-Type Stars')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = "main_sequence_lifetime_comparison.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved as {output_path}")
