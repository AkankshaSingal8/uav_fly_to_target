import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "MSE_marker_x0.4_y0_goal3.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
df.head()

# Plot the MSE values over time
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['MSE'], marker='o', linestyle='-')
plt.xlabel('Timestamp')
plt.ylabel('MSE')
plt.title('MSE over Time')
plt.grid(True)
plt.show()
