#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# File to store MSE loss values
CSV_FILE_PATH = "mse_loss_values.csv"

# Initialize the CSV file and write the header if it doesn't exist
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "MSE_Loss"])  # Write the header

IMAGE_SHAPE = (144, 256, 3)  # Target shape for resizing images
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])  # OpenCV expects (width, height)
image_pth = "goal_image.png"

# Load the goal image using OpenCV
goal_image = cv2.imread(image_pth)

if goal_image is None:
    raise FileNotFoundError(f"Goal image at path '{image_pth}' could not be loaded. Check the file path.")

# Resize the goal image to match the target shape
goal_image = cv2.resize(goal_image, IMAGE_SHAPE_CV)

# Normalize the goal image to range [0, 1] for MSE calculation
goal_image = goal_image.astype(np.float32) / 255.0

# Initialize lists to store data for real-time plotting
timestamps = []
mse_losses = []

# Initialize the plot
plt.ion()  # Turn on interactive mode for real-time plotting
fig, ax = plt.subplots()
ax.set_title("Real-Time MSE Loss")
ax.set_xlabel("Time (s)")
ax.set_ylabel("MSE Loss")
line, = ax.plot([], [], 'b-', label="MSE Loss")  # Line object for plotting
ax.legend()


# Function to update the plot
def update_plot():
    line.set_xdata(timestamps)
    line.set_ydata(mse_losses)
    ax.relim()  # Recalculate limits for axes
    ax.autoscale_view()  # Automatically scale the view to fit data
    plt.draw()
    plt.pause(0.001)  # Pause briefly to allow plot to update


# Callback function to process the incoming images
def image_callback(msg):
    bridge = CvBridge()
    try:
        # Convert to OpenCV image format
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the incoming image to match the target shape
        cv_img = cv2.resize(cv_img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

        # Normalize the incoming image to range [0, 1] for MSE calculation
        cv_img = cv_img.astype(np.float32) / 255.0

        # Compute the difference between the incoming image and the goal image
        difference = cv_img - goal_image

        # Compute the Mean Squared Error (MSE) loss
        mse_loss = np.mean(np.square(difference))
        print(f"Mean Squared Error Loss: {mse_loss}")

        # Get the current ROS time for the CSV log
        timestamp = rospy.Time.now().to_sec()

        # Append the MSE loss to the CSV file
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, mse_loss])

        # Update the data for real-time plotting
        timestamps.append(timestamp)
        mse_losses.append(mse_loss)

        # Update the real-time plot
        update_plot()

    except CvBridgeError as e:
        print(f"Error converting ROS Image message to OpenCV format: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == '__main__':
    rospy.init_node('drone_raw_image_viewer', anonymous=True)

    # Subscribe to the raw image topic
    rospy.Subscriber("/cgo3_camera/image_raw", Image, image_callback)
    print("Subscribed to /cgo3_camera/image_raw")

    try:
        plt.show(block=True)  # Keep the plot window open
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down node...")

