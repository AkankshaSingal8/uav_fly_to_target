#!/usr/bin/env python3
import sys
import rospy, math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


pub = rate = x = y = yaw = t0 = radius = omega = 0

def odom_cb(msg):
    global x, y, yaw
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    q = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])


def run_straight():
    global pub, rate
    while not rospy.is_shutdown():
        vel = Twist()
        vel.linear.x = 0.2
        pub.publish(vel)

        rate.sleep()

def run_infinity():
    global x, y, yaw, t0, pub, rate
    Kp_lin = 1.0
    Kp_ang = 3.0

    while not rospy.is_shutdown():

        t = rospy.Time.now().to_sec() - t0

        # desired parametric trajectory
        xd = math.cos(t)
        yd = math.sin(2*t)/2

        # desired velocities (for feedforward, optional)
        dxd = -math.sin(t)
        dyd = math.cos(2*t)

        # compute error
        ex = xd - x
        ey = yd - y

        # target heading
        theta_d = math.atan2(dyd, dxd)
        etheta = math.atan2(math.sin(theta_d - yaw), math.cos(theta_d - yaw))

        # control law
        v = Kp_lin * math.sqrt(ex**2 + ey**2) + 0.2
        w = Kp_ang * etheta

        # publish
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        pub.publish(cmd)

        rate.sleep()

def run_circle():
    global t0, radius, omega, rate, pub
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0

        # Position on circle (not used for cmd_vel, but good for debugging)
        x = radius * math.cos(omega * t)
        y = radius * math.sin(omega * t)

        # Velocities (derivatives)
        vx = -radius * omega * math.sin(omega * t)
        vy =  radius * omega * math.cos(omega * t)

        # Speed magnitude
        v = math.sqrt(vx**2 + vy**2)

        # For differential-drive rover: linear.x = forward speed, angular.z = curvature * velocity
        # Since it's a perfect circle: angular velocity = omega
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = omega

        pub.publish(twist)

        # Debug print
        rospy.loginfo(f"t={t:.2f} | pos=({x:.2f},{y:.2f}) | vx={vx:.2f} | vy={vy:.2f} | v={v:.2f} | w={omega:.2f}")

        rate.sleep()

if __name__ == "__main__":
    global pub, x, y, yaw, t0, rate, radius, omega
    rospy.init_node("husky__controller")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.Subscriber("/odometry/filtered", Odometry, odom_cb)
    rospy.sleep(2)
    x = y = yaw = 0.0
    t0 = rospy.Time.now().to_sec()
    rate = rospy.Rate(20)
    radius = 5
    omega = 1 
    shape = sys.argv[1]
    if(shape == "circle"):
        run_circle()
    elif(shape == "infinity"):
        run_infinity()
    else:
        run_straight()
