#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist

def sine_path_controller():
    rospy.init_node("husky_sine_path", anonymous=False)
    vel_pub = rospy.Publisher("/husky_velocity_controller/cmd_vel", Twist, queue_size=10)
    rate = rospy.Rate(20)  # 20 Hz

    A = 1.0  # x-scaling factor
    B = 1.0  # sine wave amplitude
    omega = 0.5  # angular rate of progression (rad/s)

    start_time = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - start_time
        theta = omega * t

        if theta >= 2 * math.pi:
            # Stop the robot
            stop_cmd = Twist()
            vel_pub.publish(stop_cmd)
            rospy.loginfo("Completed full sine path. Stopping.")
            break

        # Parametrize path in terms of theta (angle), not time directly
        x_theta = A * theta
        y_theta = B * math.sin(theta)

        dx_dtheta = A
        dy_dtheta = B * math.cos(theta)

        d2x_dtheta2 = 0
        d2y_dtheta2 = -B * math.sin(theta)

        # Convert derivatives w.r.t theta into derivatives w.r.t time
        dx = dx_dtheta * omega
        dy = dy_dtheta * omega

        ddx = d2x_dtheta2 * omega**2
        ddy = d2y_dtheta2 * omega**2

        # Compute linear and angular velocity
        v = math.sqrt(dx**2 + dy**2)
        if v == 0:
            omega_cmd = 0.0
        else:
            omega_cmd = (dy * ddx - dx * ddy) / (v**2)

        # Send command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega_cmd
        vel_pub.publish(cmd)

        rate.sleep()

if __name__ == "__main__":
    try:
        sine_path_controller()
    except rospy.ROSInterruptException:
        pass
