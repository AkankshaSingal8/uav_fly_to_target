#!/usr/bin/env python
import rospy
import sys
import numpy as np
from hector_uav_msgs.srv import EnableMotors, EnableMotorsRequest, EnableMotorsResponse
import actionlib
from hector_uav_msgs.msg import TakeoffAction, TakeoffGoal, TakeoffResult, TakeoffFeedback
from hector_uav_msgs.msg import HeightCommand, PositionXYCommand, VelocityXYCommand, VelocityZCommand
from geometry_msgs.msg import Twist
import time
from hector_uav_msgs.msg import PoseAction, PoseGoal,PoseResult, PoseFeedback
import math
import tf
import time
from math import cos, exp, pi
import matplotlib.pyplot as plt
import argparse

time_start = time.time()


def enable_motors(value):
    if value == True:
        print("enabling motors...")
    elif value == False:
        print("disabling motors...")
    rospy.wait_for_service("/hector/enable_motors")
    enable_motors = rospy.ServiceProxy("/hector/enable_motors", EnableMotors)
    
    try:
        service_response = enable_motors(value)
        return service_response
    except Exception as e:
        print("Error in enable_motors(): ", e) 

def takeoff():
    client = actionlib.SimpleActionClient("/hector/action/takeoff", TakeoffAction)
    client.wait_for_server()
    print(10)
    try:
        client.send_goal(10)
        #client.wait_for_result()
        print(client.get_state())
    except Exception as e:
        print("Error in takeoff(): ", e)
    return client.get_result()

def command_pose(px, py, pz, x, y, z, w):
    client = actionlib.SimpleActionClient("/hector/action/pose", PoseAction)
    client.wait_for_server()

    pose_goal = PoseGoal()
    pose_goal.target_pose.header.stamp = rospy.Time.now()
    pose_goal.target_pose.header.frame_id = "world"
    pose_goal.target_pose.pose.position.x = px
    pose_goal.target_pose.pose.position.y = py
    pose_goal.target_pose.pose.position.z = pz
    pose_goal.target_pose.pose.orientation.x = x
    pose_goal.target_pose.pose.orientation.y = y
    pose_goal.target_pose.pose.orientation.z = z
    pose_goal.target_pose.pose.orientation.w = w
    quaternion = (pose_goal.target_pose.pose.orientation.x,pose_goal.target_pose.pose.orientation.y,pose_goal.target_pose.pose.orientation.z,pose_goal.target_pose.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion) #convert quaternion to euler
    print(euler)

    #while True:
    try:
        client.send_goal(pose_goal)
        #client.wait_for_result()
    except Exception as e:
        print("Error in command_pose(): ", e)
    return client.get_result()
    
def Controller(px, py, pz, x, y, z, w):
    rospy.init_node("Hector_Init", anonymous=False)
    enable_motor_result = enable_motors(True)
    print(enable_motor_result)

    takeoff_result = takeoff()
    print(takeoff_result)

    pose_result = command_pose(px, py, pz, x, y, z, w)
    print(pose_result)

    try:
	rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Control Hector UAV pose via command-line arguments.")
    parser.add_argument("px", type=float, help="Target position x-coordinate")
    parser.add_argument("py", type=float, help="Target position y-coordinate")
    parser.add_argument("pz", type=float, help="Target position z-coordinate")
    parser.add_argument("x", type=float, help="Orientation quaternion x")
    parser.add_argument("y", type=float, help="Orientation quaternion y")
    parser.add_argument("z", type=float, help="Orientation quaternion z")
    parser.add_argument("w", type=float, help="Orientation quaternion w")
    
    args = parser.parse_args()

    try:
        Controller(args.px, args.py, args.pz, args.x, args.y, args.z, args.w)
    except rospy.ROSInterruptException:
        pass
    
