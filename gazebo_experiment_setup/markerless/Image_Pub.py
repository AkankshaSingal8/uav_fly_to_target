#!/usr/bin/env python3
import rospy
import cv2
import sys
import time
from std_msgs.msg import Float32MultiArray
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# import apriltag

# Default camera parameters
# update rate = 30Hz
# resolution = 640x480
# image format = L8
# horizontal fov = 100

global centres_data

centres_data = Float32MultiArray()


def converter(data):
    # pt_star = np.array([364, 293, 276, 293, 276, 205, 364, 205])
    
    try:
        # Convert ROS Image message to OpenCV format
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")      

        # Display the processed camera feed
        cv2.imshow("Downward Camera Feed", cv_image)
        cv2.waitKey(1)

       
    except CvBridgeError as e:
        print(e)


def camera_feed():
    rospy.init_node('node1_feed', anonymous=True)
    rospy.Subscriber("/hector/downward_cam/camera/image", Image, converter)
    
    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        
        rate.sleep()
    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    try:
        camera_feed()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()

