#!/usr/bin/env python3
import rospy
import cv2
import sys
import time
from std_msgs.msg import Float32MultiArray
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import apriltag

# Default camera parameters
# update rate = 30Hz
# resolution = 640x480
# image format = L8
# horizontal fov = 100

global centres_data

centres_data = Float32MultiArray()
goal_height = 5

def converter(data):
    global goal_height
    # pt_star = np.array([364, 293, 276, 293, 276, 205, 364, 205])
    if goal_height == 3:
        pt_star = np.array([364, 293, 276, 293,276, 205,364, 205]) 
    elif goal_height == 4:
        pt_star = np.array([351, 278, 289, 278, 289, 215, 351, 215]) #height 4
    elif goal_height == 5:
        pt_star = np.array([344, 269, 296, 269, 296, 221, 344, 221]) #Desired pixel points location height 5
    
    try:
        # Convert ROS Image message to OpenCV format
        cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
        options = apriltag.DetectorOptions(families='tag36h11')
        
        # Initialize the AprilTag detector
        detector = apriltag.Detector(options)
        img = CvBridge().imgmsg_to_cv2(data, "mono8")
        result = detector.detect(img)

        # If AprilTags are detected
        if len(result) > 0:
            for r in result:
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))

                # Draw lines connecting the corners of the detected AprilTag
                cv2.line(cv_image, ptA, (pt_star[0], pt_star[1]), (0, 255, 0), 2)
                cv2.line(cv_image, ptB, (pt_star[2], pt_star[3]), (0, 255, 0), 2)
                cv2.line(cv_image, ptC, (pt_star[4], pt_star[5]), (0, 255, 0), 2)
                cv2.line(cv_image, ptD, (pt_star[6], pt_star[7]), (0, 255, 0), 2)

                # Draw the center (x, y)-coordinates of the AprilTag
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                cv2.circle(cv_image, (cX, cY), 5, (0, 0, 255), -1)

                # Highlight the tag corners and reference points
                cv2.circle(cv_image, ptA, 3, (255, 0, 0), -1)
                cv2.circle(cv_image, ptB, 3, (255, 0, 0), -1)
                cv2.circle(cv_image, ptC, 3, (255, 0, 0), -1)
                cv2.circle(cv_image, ptD, 3, (255, 0, 0), -1)
                cv2.circle(cv_image, ((pt_star[0] + pt_star[2] + pt_star[4] + pt_star[6]) // 4,
                                      (pt_star[1] + pt_star[3] + pt_star[5] + pt_star[7]) // 4), 5, (0, 255, 0), -1)
                cv2.circle(cv_image, (pt_star[0], pt_star[1]), 3, (0, 255, 0), -1)
                cv2.circle(cv_image, (pt_star[2], pt_star[3]), 3, (0, 255, 0), -1)
                cv2.circle(cv_image, (pt_star[4], pt_star[5]), 3, (0, 255, 0), -1)
                cv2.circle(cv_image, (pt_star[6], pt_star[7]), 3, (0, 255, 0), -1)

        # Display the processed camera feed
        cv2.imshow("Downward Camera Feed", cv_image)
        cv2.waitKey(1)

        # Publish the AprilTag corner coordinates
        centres_data.data = np.array([r.corners[0][0], r.corners[0][1],
                                      r.corners[1][0], r.corners[1][1],
                                      r.corners[2][0], r.corners[2][1],
                                      r.corners[3][0], r.corners[3][1]])
        # print(centres_data.data)

    except CvBridgeError as e:
        print(e)


def camera_feed():
    rospy.init_node('node1_feed', anonymous=True)
    rospy.Subscriber("/hector/downward_cam/camera/image", Image, converter)
    aprCentre_pub = rospy.Publisher("/apriltag_centres", Float32MultiArray, queue_size=4)
    rate = rospy.Rate(30)  # 30Hz
    while not rospy.is_shutdown():
        aprCentre_pub.publish(centres_data)
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

