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


    

# Callback for odometry data (height & position)
def odometry_callback(data):  
    

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

    print(f"Current pos: {current_position} current orientation: {current_orientation} ")
    




# Main function
if __name__ == '__main__':
    try:
        # Initialize ROS node
        rospy.init_node('read pos', anonymous=True)

        # Subscribers
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)


        # Define loop rate
        rate = rospy.Rate(20)  # 20 Hz

        
        # Main loop
        while not rospy.is_shutdown():
            
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

