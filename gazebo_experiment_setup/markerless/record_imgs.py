#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import csv
import os
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Image and recording settings
IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
OUTPUT_DIR = "output_images"
CSV_FILE = "positions.csv"

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Global variables
current_height = 0.0  # Track current UAV height
recording_active = False  # Flag to start recording
bridge = CvBridge()
CV_IMAGE = None
current_velocity = Twist()  # Store latest velocity command
count = 1
# Open CSV file for recording positions & velocity
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["count", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"])  # CSV Header

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
            count, x, y, z, 
            current_velocity.linear.x, current_velocity.linear.y, current_velocity.linear.z, 
            current_velocity.angular.x, current_velocity.angular.y, current_velocity.angular.z
        ])
        csv_file.flush()

# Callback for image data
def image_callback(data):
    global bridge, CV_IMAGE, recording_active, count

    try:
        # Convert ROS Image to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # Save image if recording is active
        if recording_active:
            timestamp = rospy.Time.now().to_sec()
            image_filename = os.path.join(OUTPUT_DIR, f"Image{count}.png")
            cv2.imwrite(image_filename, cv_image)
            count += 1

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

