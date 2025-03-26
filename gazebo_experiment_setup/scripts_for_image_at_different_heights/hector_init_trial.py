 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])

not_saved = False




current_height = 0.0  # Global variable to track the current height
current_orientation = 0.0
bridge = CvBridge() 
# Define the Twist message for velocity commands

# Callback to update the current height from the odometry topic
def odometry_callback(data):
    global current_height
    
    current_height = data.pose.pose.position.z  # Extract Z position (height)
    
    
def image_callback(data):
    global bridge
    global current_height
    global not_saved
    
    try:
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        #cv_img = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 
        
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
    if current_height >= 3 and not_saved != True:
    	cv2.imwrite("./goal_marker_markerless_height3_z1.png",cv_image)
    	not_saved = True
    

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
            

            # Sleep to maintain the loop rate
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

