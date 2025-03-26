#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import time
from mavros_msgs.srv import CommandBool, SetMode


import csv



current_state = State()
current_pose = PoseStamped()


vel_msg = Twist()



def state_cb(msg):
    global current_state
    current_state = msg

def position_cb(pose):
    global current_pose
    current_pose = pose
    
    
    px = current_pose.pose.position.x
    py = current_pose.pose.position.y
    pz = current_pose.pose.position.z
    
    timestamp = time.time()

    # Data to save
    data = [timestamp, px, py, pz]

	# Save to CSV
    csv_file = './positions.csv'
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the file is empty
        if file.tell() == 0:
            writer.writerow(["Timestamp", "px", "py", "pz"])
        writer.writerow(data)



    #rospy.loginfo("got velocity")
    rospy.loginfo(f"Current Position: x={px}, y={py}, z={pz} ")
    rospy.loginfo(f" velocity: vx={vx}, vy={vy}, vz={vz}, omega_z={omega_z}")
    VEL = True
    COUNT += 1

    
if __name__ == "__main__":
    rospy.init_node("position_dump")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    local_position_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, position_cb)

    

