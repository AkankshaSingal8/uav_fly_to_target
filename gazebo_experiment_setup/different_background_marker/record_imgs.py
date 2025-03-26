#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os


IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])



rospy.loginfo("current working dir")
rospy.loginfo(os.getcwd())

goal_image_file = './marker_goal_images/goal_marker_height3.png'
goal_image = cv2.imread(goal_image_file)
goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 



current_height = 0.0  # Global variable to track the current height
bridge = CvBridge() 

model_vel = Twist()
CV_IMAGE = None


# Callback to update the current height from the odometry topic
def odometry_callback(data):  
    global current_height
    current_height = data.pose.pose.position.z  # Extract Z position (height)
    
def image_callback(data):
    global bridge
    global CV_IMAGE
    global model_vel
    
    try:
        # Convert the ROS Image message to a CV2 image
        
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        #print(cv_image.shape)
        #cv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("output.png", cv_image)
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

        cv2.imshow("Camera Image", cv_image)
        cv2.waitKey(0)
        cv_img = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 
        
        CV_IMAGE = cv_img
        #cv2.imwrite("output.png", cv_img)
        
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))


if __name__ == '__main__':
 
    try:
        # Initialize the ROS node
        rospy.init_node('hector_velocity_controller', anonymous=True)

        

        # Subscriber to the Odometry topic to get the current height
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        
        rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)

        # Define rate of publishing
        rate = rospy.Rate(20)  # 20 Hz

        velocity_cmd = Twist()

        # Main control loop
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

