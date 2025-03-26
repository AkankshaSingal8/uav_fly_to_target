import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





def plot_3d_trajectory(data):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['Px'], data['Py'], data['Pz'], label='3D Trajectory', color='b')
    ax.set_xlabel('Position X (px)')
    ax.set_ylabel('Position Y (py)')
    ax.set_zlabel('Position Z (pz)')
    ax.set_title('3D Trajectory')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Path to the CSV file
    csv_file = "./odometry_data.csv"  # Replace with your actual CSV file path

    # Load data
    data = pd.read_csv(csv_file)
    #print(f" Max vx: {max(data['vx'])}, Min vx: {min(abs(data['vx']))}")
    #print(f" Max vy: {max(data['vy'])}, Min vy: {min(abs(data['vy']))}")
    #print(f" Max vz: {max(data['vz'])}, Min vz: {min(abs(data['vz']))}")
    #print(f" Max omega_z: {max(data['omega_z'])}, Min omega_z: {min(abs(data['omega_z']))}")

    # Plot trajectories
    plot_3d_trajectory(data)
    #plot_velocity(data)
    #plot_angular_velocity(data)

