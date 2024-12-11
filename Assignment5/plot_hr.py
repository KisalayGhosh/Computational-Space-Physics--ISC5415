import mesa_reader as mr
import matplotlib.pyplot as plt

# Change the filename for each simulation
history_file = 'history_3M.data'  # Update to history_10M.data or history_3M.data
history = mr.MesaData(history_file)

# Extract data
log_Teff = history.log_Teff
log_L = history.log_L

# Plot HR Diagram
plt.figure()
plt.plot(log_Teff, log_L, label='25M Star')
plt.xlabel('log(Teff)')
plt.ylabel('log(L)')
plt.gca().invert_xaxis()  # HR diagrams conventionally have x-axis inverted
plt.title('Hertzsprung-Russell Diagram')
plt.legend()
plt.savefig('hr_diagram_3M.png')  # Save the plot with an appropriate name
plt.show()
