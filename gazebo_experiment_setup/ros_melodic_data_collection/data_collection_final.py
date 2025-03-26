#!/usr/bin/env python
import rospy
import sys
import os
import cv2
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
import time
import math
import tf
from math import cos, exp, pi, sin, sqrt
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import apriltag
import subprocess
import threading
import argparse


def delete_specific_rosbag(rosbag_filename):
    print("Deleting file: ", rosbag_filename)
    try:
        os.remove(rosbag_filename)
        print("Rosbag file deleted successfully.")
        
    except Exception as e:
        print("Error while deleting rosbag file: ", e)


def check_position_and_orientation(goal_position, goal_orientation, hector_process, rosbag_process, ibvs_process):

    # Set the tolerance for position and orientation matching
    position_tolerance = 0.09  # meters
    orientation_tolerance = 0.1  

    # Define a callback for the /hector/ground_truth/state topic
    def odometry_callback(data):
	#print("Entering callback")
        current_position = (
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z
        )
        current_orientation = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        )

        # Check if position and orientation are within the tolerances
	#print(current_position[0] - goal_position[0], current_position[1] - goal_position[1], current_position[2] - goal_position[2], current_orientation[0] - goal_orientation[0], current_orientation[1] - goal_orientation[1], current_orientation[2] - goal_orientation[2], current_orientation[3] - goal_orientation[3])
        if (
            abs(current_position[0] - goal_position[0]) <= 0.2 and
            abs(current_position[1] - goal_position[1]) <= position_tolerance and
            abs(current_position[2] - goal_position[2]) <= position_tolerance and
	    abs(current_orientation[0] - goal_orientation[0]) <= orientation_tolerance and
            abs(current_orientation[1] - goal_orientation[1]) <= orientation_tolerance and
            (abs(current_orientation[2] - goal_orientation[2]) <= orientation_tolerance or abs(current_orientation[2] - 1) <= orientation_tolerance)and
            abs(current_orientation[3] - goal_orientation[3]) <= orientation_tolerance
        ):
            print("Goal position and orientation reached.")
            print("Shutting down all subprocesses...")
	    time.sleep(5)
	    
            if rosbag_process.poll() is None:
                rosbag_process.terminate()
                print("rosbag recording terminated.")
	    if ibvs_process.poll() is None:
                ibvs_process.terminate()
                print("IBVS_Static.py terminated.")
	    # Terminate all subprocesses
            if hector_process.poll() is None:
                hector_process.terminate()
                print("Hector_Initialisation.py terminated.")
            

            # Delete the rosbag file
            #delete_specific_rosbag(rosbag_filename)

            # Shut down ROS node
            rospy.signal_shutdown("Goal position and orientation reached.")

    # Subscribe to the /hector/ground_truth/state topic
    rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
    print("SUbscribed to rostopic.")


def run_hector_initialisation_and_record(script_path, position, orientation, output_directory, ibvs_script, goal_position, goal_orientation):
    """
    Runs the Hector_Initialisation.py script with the given position and orientation values,
    waits for 15 seconds, records a ROS bag, and then runs the IBVS_Static.py script.

    Args:
        script_path (str): Path to the Hector_Initialisation.py script.
        position (tuple): A tuple of three floats representing the x, y, and z positions.
        orientation (tuple): A tuple of four floats representing the x, y, z, and w orientation.
        output_directory (str): Directory where the ROS bag should be recorded.
        ibvs_script (str): Path to the IBVS_Static.py script to be executed after rosbag recording.
    
    Returns:
        None
    """
    rosbag_filename = None  # To store the name of the rosbag file
    position_x, position_y, position_z = position
    orientation_x, orientation_y, orientation_z, orientation_w = orientation

    try:
        # Initialize ROS node
        rospy.init_node("hector_monitor", anonymous=True)

        # Run the Hector_Initialisation script as a background process
        hector_process = subprocess.Popen(
            ["python", script_path, str(position_x), str(position_y), str(position_z),
                str(orientation_x), str(orientation_y), str(orientation_z), str(orientation_w)]
        )
        print("Hector_Initialisation.py script started successfully.")

        # Wait for 15 seconds
        print("Waiting for 15 sec")
        time.sleep(20)

        # Run the rosbag record command as a background process
        rosbag_process = subprocess.Popen(
            ["rosbag", "record", "-a", "--output-prefix", output_directory],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        #print("rosbag recording started in directory: ", output_directory)

	
	time.sleep(5)

        # Run the IBVS_Static script as a background process
        ibvs_process = subprocess.Popen(
            ["python", ibvs_script]
        )
        print("IBVS_Static.py script executed successfully.")

        # Start monitoring the position and orientation
        check_position_and_orientation(goal_position, goal_orientation, hector_process, rosbag_process, ibvs_process)

        # Keep the script alive until ROS is shut down
        rospy.spin()

    except subprocess.CalledProcessError as e:
        print("Error occurred while running a subprocess: ", e)
        #delete_specific_rosbag(file_name)
	return
    except Exception as e:
        print("Unexpected error: ", e)
        #delete_specific_rosbag(file_name)
	return
    finally:
        # Ensure the rosbag file is deleted if the script exits unexpectedly
        #delete_specific_rosbag(file_name)
	return


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hector Initialisation and Record Data.")
    
    # Add arguments for position
    parser.add_argument("pos_x", type=float, help="X coordinate of the position")
    parser.add_argument("pos_y", type=float, help="Y coordinate of the position")
    parser.add_argument("pos_z", type=float, help="Z coordinate of the position")
    
    # Add arguments for orientation (quaternion)
    parser.add_argument("orient_x", type=float, help="X coordinate of the orientation (quaternion)")
    parser.add_argument("orient_y", type=float, help="Y coordinate of the orientation (quaternion)")
    parser.add_argument("orient_z", type=float, help="Z coordinate of the orientation (quaternion)")
    parser.add_argument("orient_w", type=float, help="W coordinate of the orientation (quaternion)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Extract position and orientation from arguments
    position = (args.pos_x, args.pos_y, args.pos_z)
    orientation = (args.orient_x, args.orient_y, args.orient_z, args.orient_w)
    script_path = "Hector_Initialisation_updated.py"
    #position = (1.0, 1.0, 8.0)  # Goal position
    #orientation = (0.0, 0.0, 1.0, 0.0)  # Goal orientation (quaternion)
    goal_position = (0.4, 0.0, 5.0)
    goal_orientation = (0.0, 0.0, -1.0, 0.0)
    output_directory = "height5_trial/"
    ibvs_script = "IBVS_Static.py"

    run_hector_initialisation_and_record(script_path, position, orientation, output_directory, ibvs_script, goal_position, goal_orientation)

    


