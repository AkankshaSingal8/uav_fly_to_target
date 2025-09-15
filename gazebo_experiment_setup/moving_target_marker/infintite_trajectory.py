#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PurePursuitTracker:
    def __init__(self):
        """
        Initializes the Pure Pursuit tracker node.
        """
        # --- ROS Setup ---
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.path_pub = rospy.Publisher("/rover_path", Path, queue_size=10)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_cb)

        # --- Robot State ---
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.has_odom = False

        # --- Path Visualization ---
        self.rover_path = Path()
        # Set the frame_id to your odometry frame (e.g., "odom", "map")
        self.rover_path.header.frame_id = "odom"

        # --- Pure Pursuit Parameters ---
        self.lookahead_distance = 0.8
        self.linear_velocity = 0.2

        # --- Trajectory Parameters ---
        self.scale = 2.0
        self.path_resolution = 200
        self.path_points = []
        self.target_idx = 0

        # Generate the reference trajectory
        self.generate_path()

    def odom_cb(self, msg):
        """
        Callback function to receive and update the robot's current pose.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.th = yaw
        self.has_odom = True

    def generate_path(self):
        """
        Generates a list of waypoints for a scaled infinity symbol.
        """
        self.path_points = []
        for i in range(self.path_resolution + 1):
            t = 2 * math.pi * (i / self.path_resolution)
            x = self.scale * math.cos(t)
            y = self.scale * 0.5 * math.sin(2 * t)
            self.path_points.append((x, y))

    def find_lookahead_point(self):
        """
        Finds the target point on the path for the robot to pursue.
        """
        search_index = self.target_idx
        
        while True:
            px, py = self.path_points[search_index]
            dist_to_point = math.hypot(self.x - px, self.y - py)

            if dist_to_point >= self.lookahead_distance:
                self.target_idx = search_index
                return self.path_points[search_index]

            search_index = (search_index + 1) % len(self.path_points)

            if search_index == self.target_idx:
                break
        
        return self.path_points[self.target_idx]

    def update_and_publish_path(self):
        """
        Updates the rover's path with the current pose and publishes it.
        """
        current_pose = PoseStamped()
        current_pose.header.stamp = rospy.Time.now()
        current_pose.header.frame_id = self.rover_path.header.frame_id
        current_pose.pose.position.x = self.x
        current_pose.pose.position.y = self.y
        
        # Convert yaw to quaternion for the pose orientation
        q = quaternion_from_euler(0, 0, self.th)
        current_pose.pose.orientation.x = q[0]
        current_pose.pose.orientation.y = q[1]
        current_pose.pose.orientation.z = q[2]
        current_pose.pose.orientation.w = q[3]

        # Append the current pose to the path's list of poses
        self.rover_path.poses.append(current_pose)
        self.rover_path.header.stamp = rospy.Time.now()

        # Publish the complete path
        self.path_pub.publish(self.rover_path)

    def run(self):
        """
        Main control loop for the Pure Pursuit algorithm.
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if not self.has_odom:
                rate.sleep()
                continue

            # Find the lookahead point on the trajectory
            target_x, target_y = self.find_lookahead_point()

            # Calculate the angle (alpha) to the lookahead point
            angle_to_target = math.atan2(target_y - self.y, target_x - self.x)
            alpha = angle_to_target - self.th
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi

            # Calculate the required angular velocity
            angular_velocity = (2.0 * self.linear_velocity * math.sin(alpha)) / self.lookahead_distance
            
            # Publish the Twist message
            msg = Twist()
            msg.linear.x = self.linear_velocity
            msg.angular.z = angular_velocity
            self.cmd_pub.publish(msg)

            # Update and publish the path for visualization
            self.update_and_publish_path()

            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("pure_pursuit_tracker")
    tracker = PurePursuitTracker()
    try:
        tracker.run()
    except rospy.ROSInterruptException:
        pass
