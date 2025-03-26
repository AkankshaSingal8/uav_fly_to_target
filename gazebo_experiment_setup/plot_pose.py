#!/usr/bin/env python3

import rosbag
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped

# Replace this with the path to your bag file
bag_file = 'pose3xandw.bag'

# Lists to store time and position data
time = []
x_positions = []
y_positions = []
z_positions = []

# Open the bag file and read messages from the topic
with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/hector/ground_truth/state']):
        #print(msg)
        
        time.append(t.to_sec())
          
        x_positions.append(msg.pose.position.x)
        y_positions.append(msg.pose.position.y)
        z_positions.append(msg.pose.position.z)

print(x_positions, y_positions, z_positions)
# Plot x, y, z positions against time
plt.figure(figsize=(10, 6))

# Plot x position
plt.subplot(3, 1, 1)
plt.plot(time, x_positions, label='X Position', color='b')
plt.xlabel('Time [s]')
plt.ylabel('X Position [m]')
plt.grid()
plt.legend()

# Plot y position
plt.subplot(3, 1, 2)
plt.plot(time, y_positions, label='Y Position', color='g')
plt.xlabel('Time [s]')
plt.ylabel('Y Position [m]')
plt.grid()
plt.legend()

# Plot z position
plt.subplot(3, 1, 3)
plt.plot(time, z_positions, label='Z Position', color='r')
plt.xlabel('Time [s]')
plt.ylabel('Z Position [m]')
plt.grid()
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

