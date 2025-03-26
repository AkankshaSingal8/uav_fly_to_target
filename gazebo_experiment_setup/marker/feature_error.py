#!/usr/bin/env python
import rospy
import csv
import sys
import cv2
import numpy as np
import time
import math
import tf
from math import cos, sin, pi
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import apriltag

# Global variables
V_Q = np.zeros(6)
euler_hector = np.zeros(3)
pos_hector = np.zeros(3)
ang_rate = np.zeros(3)
lin_vel = np.zeros(3)

# Open CSV file
goal_height = 4
CSV_FILE = f"./new_results_with_orientation/ibvs_feature_error_marker_x1_y1_z{goal_height + 3}_goal{goal_height}.csv"
csv_file = open(CSV_FILE, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "e_v_x", "e_v_y", "e_v_a", "e_alpha"])  # CSV headers
recording_active = False  


def R_x(theta):
    return np.array([[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]])

def R_y(theta):
    return np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

def R_z(theta):
    return np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])

def Virtual(s, R_virtual):
    global goal_height
    cu, cv, f = 320.00, 240.00, 269.00  # Camera parameters
    # pt_star = np.array([344, 269, 296, 269, 296, 221, 344, 221]) # height 3
    if goal_height == 3:
        pt_star = np.array([364, 293, 276, 293,276, 205,364, 205]) 
    elif goal_height == 4:
        pt_star = np.array([351, 278, 289, 278, 289, 215, 351, 215]) #height 4
    elif goal_height == 5:
        pt_star = np.array([344, 269, 296, 269, 296, 221, 344, 221]) #Desired pixel points location height 5
    
    s_star = pt_star - np.array([320, 240, 320, 240, 320, 240, 320, 240])

    # Virtual Plane Conversion
    p_bar = [np.array([[s[i]-cu], [s[i+1]-cv], [f]]) for i in range(0, 8, 2)]
    p_bar_star = [np.array([[s_star[i]], [s_star[i+1]], [f]]) for i in range(0, 8, 2)]

    vs = np.array([
        (f / np.dot(R_virtual[2, :], p_bar[i])) * np.dot(R_virtual[0, :], p_bar[i]) for i in range(4)
    ] + [
        (f / np.dot(R_virtual[2, :], p_bar[i])) * np.dot(R_virtual[1, :], p_bar[i]) for i in range(4)
    ])

    vs_star = np.array([
        (f / np.dot(R_virtual[2, :], p_bar_star[i])) * np.dot(R_virtual[0, :], p_bar_star[i]) for i in range(4)
    ] + [
        (f / np.dot(R_virtual[2, :], p_bar_star[i])) * np.dot(R_virtual[1, :], p_bar_star[i]) for i in range(4)
    ])

    #change height
    h_d = goal_height - 0.78
    x_g, y_g = np.mean(vs[:4]), np.mean(vs[4:])
    x_g_star, y_g_star = np.mean(vs_star[:4]), np.mean(vs_star[4:])

    a = sum(np.square(vs[i] - x_g) + np.square(vs[i+4] - y_g) for i in range(4))
    a_d = sum(np.square(vs_star[i] - x_g_star) + np.square(vs_star[i+4] - y_g_star) for i in range(4))

    a_n = (h_d) * np.sqrt(a_d / a)
    x_n, y_n = (a_n / f) * x_g, (a_n / f) * y_g
    s_v = np.array([x_n, y_n, a_n])
    s_v_star = np.array([[2.52490101e-07,   7.42749113e-02,  h_d]])

    e_v = s_v - s_v_star
    e_alpha = np.arctan2(vs[1] - vs[5], vs[0] - vs[4]) - np.arctan2(vs_star[1] - vs_star[5], vs_star[0] - vs_star[4])

    return e_v, e_alpha

def converter(data):
    global V_Q, euler_hector, csv_writer, recording_active

    cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
    img_gray = CvBridge().imgmsg_to_cv2(data, "mono8")
    result = detector.detect(img_gray)

    if not result:
        rospy.logwarn("No AprilTag detected.")
        return

    s = np.array([corner for r in result for corner in r.corners.flatten()])

    cu, cv, f, k_1, k_2 = 320.00, 240.00, 269.00, 0.40, 0.40
    R_C_B = R_x(pi) @ R_z(pi / 2)
    R_euler = R_y(euler_hector[1]) @ R_x(euler_hector[0])
    R_virtual = R_C_B.T @ R_euler @ R_C_B

    e_v, e_alpha = Virtual(s, R_virtual)

    R_V_I = R_z(euler_hector[2]) @ R_x(pi) @ R_z(pi / 2)

    # Control
    V_c_virtual = k_1 * e_v[0]
    V_c_global = R_V_I @ V_c_virtual
    V_x_b = cos(euler_hector[2]) * V_c_global[0] + sin(euler_hector[2]) * V_c_global[1]
    V_y_b = -sin(euler_hector[2]) * V_c_global[0] + cos(euler_hector[2]) * V_c_global[1]
    w_z = -k_2 * e_alpha

    V_Q = np.array([V_x_b, V_y_b, V_c_global[2], 0, 0, w_z])

    # **Logging to CSV**
    if recording_active:
        timestamp = rospy.Time.now().to_sec()
        csv_writer.writerow([timestamp, e_v[0], e_v[1], e_v[2], e_alpha])
        csv_file.flush()

        rospy.loginfo(f"Logged: e_v={e_v}, e_alpha={e_alpha}")

def Orient_conversion(data):
    global pos_hector, euler_hector, recording_active, goal_height
    pos_hector = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
    quaternion = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
    euler_hector = tf.transformations.euler_from_quaternion(quaternion)
   

    current_position = (
        data.pose.pose.position.x,
        data.pose.pose.position.y,
        data.pose.pose.position.z
    )

    # Start recording once height is >= 6 meters
    if current_position[2] >= (goal_height + 3):
        recording_active = True
        print("Recording Started")

def Controller():
    rospy.init_node("IBVS_Control", anonymous=False)
    rospy.Subscriber("/hector/ground_truth/state", Odometry, Orient_conversion)
    rospy.Subscriber("/hector/downward_cam/camera/image", Image, converter)
    
    rate = rospy.Rate(20)  # 20 Hz
    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        csv_file.close()  # Close the CSV file properly

if __name__ == "__main__":
    try:
        Controller()
    except rospy.ROSInterruptException:
        pass
