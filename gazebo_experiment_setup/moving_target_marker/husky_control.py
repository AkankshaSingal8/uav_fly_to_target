#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry

def Controller():
    rospy.init_node("Husky_Control", anonymous=False)
    husky_vel_pub = rospy.Publisher("/husky_velocity_controller/cmd_vel", Twist, queue_size=5)
    rate = rospy.Rate(20)  # 30hz
    while not rospy.is_shutdown():
        V_lin = 0.2
        angular = 0.2  
        # V_ang = 0.2*sin(0.1*time_now) 
        vel_cmd1 = Twist()
        vel_cmd1.linear.x = 0.2
        vel_cmd1.linear.y = 0
        vel_cmd1.linear.z = 0
        vel_cmd1.angular.x = 0
        vel_cmd1.angular.y = 0
        vel_cmd1.angular.z = -0.2
        husky_vel_pub.publish(vel_cmd1)
        rate.sleep()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    try:
        Controller()
    except rospy.ROSInterruptException:
        pass

