#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

# Define global variables
goal_height = 3.0
z_init = 3.0
reached_goal = False
image_saved = False
current_height = 0.0
bridge = CvBridge()
CV_IMAGE = None
IMAGE_SHAPE = (240, 320)  # [height, width]

# Velocity command
model_vel = Twist()
MODEL_MODE = False

def odometry_callback(data):
    global current_height
    current_height = data.pose.pose.position.z

def image_callback(data):
    global bridge, CV_IMAGE, model_vel, reached_goal, image_saved

    try:
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        CV_IMAGE = cv_image

        # Save the image if the drone has reached the target height and hasn't saved yet
        if reached_goal and not image_saved:
            save_path = os.path.join(os.getcwd(), "snapshot_at_z3.jpg")
            cv2.imwrite(save_path, CV_IMAGE)
            rospy.loginfo(f"[IMAGE SAVED] at height {current_height:.2f} m -> {save_path}")
            image_saved = True

    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

if __name__ == '__main__':
    try:
        rospy.init_node('hector_velocity_controller', anonymous=True)
        vel_publisher = rospy.Publisher("/hector/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)

        rate = rospy.Rate(20)  # 20 Hz
        velocity_cmd = Twist()

        while not rospy.is_shutdown():
            if current_height < (z_init - 0.1) and not MODEL_MODE:
                velocity_cmd.linear.z = 1.0
                vel_publisher.publish(velocity_cmd)
            elif current_height >= z_init and not reached_goal:
                reached_goal = True
                velocity_cmd.linear.z = 0.0
                vel_publisher.publish(velocity_cmd)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
