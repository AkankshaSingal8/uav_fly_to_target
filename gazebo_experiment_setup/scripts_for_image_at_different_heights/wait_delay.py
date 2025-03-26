#!/usr/bin/env python

import rospy
import time

if __name__ == "__main__":
    rospy.init_node("wait_5_seconds")
    rospy.loginfo("Waiting for 5 seconds before launching the drone...")
    time.sleep(5)
    rospy.loginfo("Done waiting. Proceeding with launch.")

