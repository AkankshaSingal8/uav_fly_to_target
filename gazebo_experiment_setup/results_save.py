#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import csv
import os
import numpy as np
import cv2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Image and recording settings
IMAGE_SHAPE = (144, 256, 3)  # Target image size (height, width, channels)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])  # OpenCV shape (width, height)
# OUTPUT_DIR = "output_images"
CSV_FILE = "result_marker_x1_y1_goal3.csv"

# Ensure output directory exists
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# Load and preprocess goal image
GOAL_IMAGE_FILE = "./marker_goal_images/goal_marker_height3.png"
goal_image = cv2.imread(GOAL_IMAGE_FILE)
goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
goal_image = goal_image.astype(np.float32) / 255.0  # Normalize to [0,1]

# Global variables
current_height = 0.0  # Track UAV height
recording_active = False  # Flag to start recording
bridge = CvBridge()
current_velocity = Twist()  # Store latest velocity command
count = 0  # Image counter

# Open CSV file for recording data
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "mse_loss"])  # CSV Header

# Callback for odometry data (height & position)
def odometry_callback(data):  
    global current_height, recording_active, csv_writer, current_velocity

    # Get UAV position
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    z = data.pose.pose.position.z
    current_height = z

    # Start recording once height is >= 6 meters
    if current_height >= 6:
        recording_active = True

    # If recording is active, log the position & velocity
    if recording_active:
        timestamp = rospy.Time.now().to_sec()
        csv_writer.writerow([
            timestamp, x, y, z, 
            current_velocity.linear.x, current_velocity.linear.y, current_velocity.linear.z, 
            current_velocity.angular.x, current_velocity.angular.y, current_velocity.angular.z,
            "N/A"  
        ])
        csv_file.flush()

# Callback for image data
def image_callback(data):
    global bridge, recording_active, count, csv_writer

    try:
        # Convert ROS Image to OpenCV format
        cv_img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # Resize and normalize incoming image
        cv_img = cv2.resize(cv_img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        cv_img = cv_img.astype(np.float32) / 255.0  # Normalize to [0,1]

        # Compute Mean Squared Error (MSE) Loss
        mse_loss = np.mean((cv_img - goal_image) ** 2)

        # Save image if recording is active
        if recording_active:
            timestamp = rospy.Time.now().to_sec()
            image_filename = os.path.join(OUTPUT_DIR, f"Image{count}.png")
            cv2.imwrite(image_filename, (cv_img * 255).astype(np.uint8))  # Convert back to uint8 for saving
            count += 1

            # Update the latest row in CSV with MSE loss
            csv_writer.writerow([
                timestamp, "N/A", "N/A", "N/A",  
                "N/A", "N/A", "N/A", 
                "N/A", "N/A", "N/A",
                mse_loss  # MSE loss value
            ])
            csv_file.flush()

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

# Callback for velocity data
def velocity_callback(data):
    global current_velocity
    current_velocity = data  # Store the latest velocity command

# Main function
if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('hector_velocity_controller', anonymous=True)

        # Subscribers
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)
        rospy.Subscriber("/hector/cmd_vel", Twist, velocity_callback)  # Subscribe to velocity topic

        # Define loop rate
        rate = rospy.Rate(20)  # 20 Hz

        rospy.loginfo("Starting data recording... Waiting for height >= 6m")

        # Main loop
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        csv_file.close()  # Ensure the CSV file is closed properly

