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
goal_height = 4
z_init = 7
CSV_FILE = f"./new_results_with_orientation/moving_linear_result_marker_x0.4_y0_z{z_init}_goal{goal_height}_husky0.05.csv"
# MSE_CSV_FILE = "./new_results_with_orientation/MSE_markerless_x0.4_y0_z8_goal6.csv"

# Ensure output directory exists
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# Load and preprocess goal image
GOAL_IMAGE_FILE = f"./marker_goal_images/goal_marker_height{goal_height}.png"
goal_image = cv2.imread(GOAL_IMAGE_FILE)
goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
goal_image = goal_image.astype(np.float32)

# Global variables
current_height = 0.0  # Track UAV height
recording_active = False  # Flag to start recording
bridge = CvBridge() 
current_velocity = Twist()  # Store latest velocity command
count = 0  # Image counter
current_position = None
current_orientation = None
husky_position = None

# Open CSV file for recording data
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "vx", "vy", "vz", "wx", "wy", "wz", "Husky_x", "Husky_y", "Husky_z"])

# Open CSV file for recording data
# mse_csv_file = open(MSE_CSV_FILE, "w", newline="")
# MSE_csv_writer = csv.writer(mse_csv_file)
# MSE_csv_writer.writerow(["timestamp", "MSE"]) 

# Define goal position and orientation conditions
GOAL_POSITION = (0.4, 0.0, 3.0)  # Target (x, y, z)


GOAL_ORIENTATIONS =   (0, 0, -1, 0)

tolerance = 0.2

# Function to check if the UAV has reached the goal position

    
    

# Callback for odometry data (height & position)
def odometry_callback(data):  
    global current_position, current_orientation, current_height, recording_active, csv_writer, current_velocity, husky_position

    current_position = (
        data.pose.pose.position.x,
        data.pose.pose.position.y,
        data.pose.pose.position.z
    )

    # Get UAV orientation (quaternion)
    current_orientation = (
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z,
        data.pose.pose.orientation.w
    )

    # Start recording once height is >= 6 meters
    if current_position[2] >= z_init:
        recording_active = True
        print("Recording Started")

    # If recording is active, log the position & velocity
    if recording_active:
        timestamp = rospy.Time.now().to_sec()
        csv_writer.writerow([
            timestamp, *current_position, *current_orientation,
            current_velocity.linear.x, current_velocity.linear.y, current_velocity.linear.z, 
            current_velocity.angular.x, current_velocity.angular.y, current_velocity.angular.z,
            *(husky_position if husky_position else (None, None, None))
        ])
        csv_file.flush()
    
# Husky Odometry Callback
def husky_odometry_callback(data):
    global husky_position
    husky_position = (
        data.pose.pose.position.x,
        data.pose.pose.position.y,
        data.pose.pose.position.z
    )
    

# Callback for image data
# def image_callback(data):
#     global bridge, recording_active, count, MSE_csv_writer

#     try:
#         # Convert ROS Image to OpenCV format
#         cv_img = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

#         # Resize and normalize incoming image
#         cv_img = cv2.resize(cv_img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
#         cv_img = cv_img.astype(np.float32) 

#         # Compute Mean Squared Error (MSE) Loss
#         mse_loss = np.mean((cv_img - goal_image) ** 2)

#         # Save image if recording is active
#         if recording_active:
#             timestamp = rospy.Time.now().to_sec()
            
#             # Update the latest row in CSV with MSE loss
#             MSE_csv_writer.writerow([
#                 timestamp, mse_loss
#             ])
#             mse_csv_file.flush()

#     except CvBridgeError as e:
#         rospy.logerr("CvBridge Error: {0}".format(e))

# Callback for velocity data
def velocity_callback(data):
    global current_velocity
    current_velocity = data  # Store the latest velocity command

# Main function
if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('result_recorder', anonymous=True)

        # Subscribers
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        # rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)
        rospy.Subscriber("/hector/cmd_vel", Twist, velocity_callback)  # Subscribe to velocity topic
        rospy.Subscriber("/hector/cmd_vel", Twist, velocity_callback)
        rospy.Subscriber("/husky_velocity_controller/odom", Odometry, husky_odometry_callback)  # Husky's odometry

        # Define loop rate
        rate = rospy.Rate(20)  # 20 Hz

        rospy.loginfo("Starting data recording... Waiting for height ")

        # Main loop
        while not rospy.is_shutdown():
            
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        csv_file.close()  # Ensure the CSV file is closed properly
        # mse_csv_file.close()

