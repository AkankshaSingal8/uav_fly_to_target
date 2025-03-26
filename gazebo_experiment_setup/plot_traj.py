import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Path to the CSV file
CSV_FILE = 'odometry_data.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(CSV_FILE)

# Ensure the file has columns Px, Py, and Pz
if not {'Px', 'Py', 'Pz'}.issubset(data.columns):
    raise ValueError("CSV file must contain columns: Px, Py, Pz")

# Extract x, y, and height (z) values
x = data['Px']
y = data['Py']
z = data['Pz']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
sc = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

# Add colorbar for height (z-axis) representation
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label('Height (Pz)')

# Set labels and title
ax.set_xlabel('Px (X)')
ax.set_ylabel('Py (Y)')
ax.set_zlabel('Pz (Height)')
ax.set_title('3D Trajectory Plot')

# Show the plot
plt.show()

