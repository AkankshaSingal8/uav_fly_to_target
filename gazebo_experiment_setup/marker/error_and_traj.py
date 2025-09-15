#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Constants
IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
CSV_FILE = 'odometry_data.csv'

# Globals
current_position = {"Px": 0.0, "Py": 0.0, "Pz": 0.0}
CV_IMAGE = None
goal_image = None
bridge = CvBridge()
current_height = 0
# Initialize MSE tracking
mse_loss_list = []
time_steps = []
RECORD = False

# Odometry callback
def odometry_callback(data):
    global current_position
    global current_height
    
    current_height = data.pose.pose.position.z  # Extract Z position (height)
    
    # Extract x, y, and z positions
    current_position["Px"] = data.pose.pose.position.x
    current_position["Py"] = data.pose.pose.position.y
    current_position["Pz"] = data.pose.pose.position.z

def save_csv():
    global RECORD
    # Save to CSV
    if RECORD:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_position["Px"], current_position["Py"], current_position["Pz"]])

# Image callback
def image_callback(data):
    global bridge
    global CV_IMAGE
    global goal_image

    try:
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        cv_img = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
        CV_IMAGE = cv_img

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def plot_image_error():
    global CV_IMAGE
    global goal_image
    global RECORD
    
    if RECORD:
        # Compute MSE loss between the current image and the goal image
        if goal_image is not None:
            mse_loss = np.mean((CV_IMAGE.astype("float32") - goal_image.astype("float32")) ** 2)
            mse_loss_list.append(mse_loss)
            time_steps.append(rospy.get_time())

            # Plot MSE loss
            plt.clf()
            plt.plot(time_steps, mse_loss_list, label="MSE Loss")
            plt.xlabel("Time")
            plt.ylabel("MSE Loss")
            plt.title("MSE Loss vs. Time")
            plt.legend()
            plt.pause(0.01)  # Allows real-time plotting

if __name__ == '__main__':
    try:
        # Initialize the ROS node
        rospy.init_node('fly_to_target', anonymous=True)

        # Check if the CSV file exists; if not, create it with headers
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Px", "Py", "Pz"])

        # Load the goal image
        goal_image_file = './marker_goal_images/goal_marker_height5.png'
        goal_image = cv2.imread(goal_image_file)
        if goal_image is None:
            rospy.logerr("Failed to load goal image at: {}".format(goal_image_file))
            exit()
        goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

        # Subscribers
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)

        # Define rate of publishing
        rate = rospy.Rate(20)  # 20 Hz

        # Initialize matplotlib for real-time plotting
        plt.ion()
        plt.figure()

        # Main control loop
        while not rospy.is_shutdown():
            # Save odometry to CSV if height threshold is met
            if current_height >= 6:
                RECORD = True
            save_csv()
            plot_image_error()
            
            # Sleep to maintain the loop rate
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # Close matplotlib properly
        plt.ioff()
        plt.close()

