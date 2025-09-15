#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from typing import Iterable, Dict
import tensorflow as tf
import kerasncp as kncp
from kerasncp.tf import LTCCell, WiredCfcCell
from tensorflow.python.keras.models import Functional
from tensorflow import keras
import numpy as np
from matplotlib.image import imread
from keras_models import generate_ncp_model
import os
import csv
from tensorflow.keras import Input, Model

def generate_hidden_list(model: Functional, return_numpy: bool = True):
    """
    Generates a list of tensors that are used as the hidden state for the argument model when it is used in single-step
    mode. The batch dimension (0th dimension) is assumed to be 1 and any other dimensions (seq len dimensions) are
    assumed to be 0

    :param return_numpy: Whether to return output as numpy array. If false, returns as keras tensor
    :param model: Single step functional model to infer hidden states for
    :return: list of hidden states with 0 as value
    """
    constructor = np.zeros if return_numpy else tf.zeros
    hiddens = []
    print("Length of model input shape: ", len(model.input_shape))
    if len(model.input_shape)==1:
        lool = model.input_shape[0][1:]
    else:
        print("model input shape: ", model.input_shape)
        lool = model.input_shape[1:]
    print("lool: ", lool)
    for input_shape in lool:  # ignore 1st output, as is this control output
        hidden = []
        for i, shape in enumerate(input_shape):
            if shape is None:
                if i == 0:  # batch dim
                    hidden.append(1)
                    continue
                elif i == 1:  # seq len dim
                    hidden.append(0)
                    continue
                else:
                    print("Unable to infer hidden state shape. Leaving as none")
            hidden.append(shape)
        hiddens.append(constructor(hidden))
    return hiddens

IMAGE_SHAPE = (144, 256, 3)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])


tf.config.set_visible_devices([], 'GPU')

rospy.loginfo("current working dir")
rospy.loginfo(os.getcwd())
root = './retrain_mix_goal_heights_diff_coreset_wscheduler0.85_seed22222_lr0.001_trainloss0.00008_epoch100.h5'

goal_height = 3
z_init = 6
goal_image_file = f'./marker_goal_images/goal_marker_height{goal_height}.png'
goal_image = cv2.imread(goal_image_file)
goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 

# CSV_FILE = f"./new_results_with_orientation/inference_time_goal{goal_height}.csv"
# csv_file = open(CSV_FILE, "w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["time"])

DEFAULT_NCP_SEED = 22222

batch_size = None
seq_len = 64
augmentation_params = None
no_norm_layer = False
single_step = True
model = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)
model.load_weights(root)

hiddens = generate_hidden_list(model= model, return_numpy=True)




current_height = 0.0  # Global variable to track the current height
bridge = CvBridge() 
# Define the Twist message for velocity commands
model_vel = Twist()
CV_IMAGE = None
MODEL_MODE = False




# def load_image(image_path):
#     img = Image.open(image_path)
#     img = img.resize(IMAGE_SHAPE_CV)
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.convert_to_tensor(img_array)
#     return img_array

def get_feature_extraction_model(mymodel):
    # Get first input (image input)
    first_input = mymodel.inputs[0]

    # Get output up to dense layer (index 9)
    last_layer_output = mymodel.layers[9].output

    # Build the new model
    model_feature_extraction_till_dense = Model(inputs=first_input, outputs=last_layer_output)
    return model_feature_extraction_till_dense

def get_output_model(mymodel):
    dense_input = Input(shape=(128,))   # for dropout_8 input
    states_input = Input(shape=(34,)) # for ltc_cell states

    # Reuse layers from original model
    dropout_out = mymodel.layers[10](dense_input)                # dropout
    ltc_out = mymodel.layers[12](dropout_out, states_input)      # ltc_cell

    # Build new model
    output_model = Model(inputs=[dense_input, states_input], outputs=ltc_out)
    return output_model

model_feature_extraction_till_dense = get_feature_extraction_model(model)
output_model = get_output_model(model)

goal_img_array = np.array(goal_image)
goal_img_array = np.expand_dims(goal_img_array, 0)
IDEAL_FEATURE_VECTOR = model_feature_extraction_till_dense.predict(goal_img_array)
CURRENT_FEATURE_VECTOR = None


# Callback to update the current height from the odometry topic
def odometry_callback(data):  
    global current_height
    current_height = data.pose.pose.position.z  # Extract Z position (height)
    
def image_callback(data):
    global bridge
    global CV_IMAGE
    global model_vel
    global IDEAL_FEATURE_VECTOR, CURRENT_FEATURE_VECTOR
    
    try:
        # Convert the ROS Image message to a CV2 image
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        
        cv_img = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0])) 
        #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        CV_IMAGE = cv_img
        current_img = np.array(cv_img)
        current_img = np.expand_dims(current_img, 0)
        CURRENT_FEATURE_VECTOR = model_feature_extraction_till_dense.predict(current_img)
        

    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def model_prediction():
    global CV_IMAGE
    global hiddens
    global model_vel
    global IDEAL_FEATURE_VECTOR, CURRENT_FEATURE_VECTOR


    feature_diff = IDEAL_FEATURE_VECTOR - CURRENT_FEATURE_VECTOR
    
    output = output_model.predict([feature_diff, *hiddens])
    
    hiddens = output[1:]
    preds = output[0][0]
    vx, vy, vz, omega_z = preds[0], preds[1], preds[2], preds[3]
    # print(f"Printing vel: {vx}, {vy}, {vz}, {omega_z}")

    # if abs(vx) <= 0.005:
    #     vx = 0.0
    
    # if abs(vy) <= 0.005:
    #     vy = 0.0
    
    # if abs(vz) <= 0.005:
    #     vz = 0.0
    
    # if abs(omega_z) <= 0.009:
    #     omega_z = 0.0
    print(f"Printing vel: {vx}, {vy}, {vz}, {omega_z}")

    # Set the velocities in the message
    model_vel.linear.x = vx
    model_vel.linear.y = vy
    model_vel.linear.z = vz
    model_vel.angular.z = omega_z
    

if __name__ == '__main__':
 
    try:
        # Initialize the ROS node
        rospy.init_node('hector_velocity_controller', anonymous=True)

        # Publisher for sending velocity commands
        vel_publisher = rospy.Publisher("/hector/cmd_vel", Twist, queue_size=1)

        # Subscriber to the Odometry topic to get the current height
        rospy.Subscriber("/hector/ground_truth/state", Odometry, odometry_callback)
        
        rospy.Subscriber("/hector/downward_cam/camera/image", Image, image_callback)

        # Define rate of publishing
        rate = rospy.Rate(20)  # 20 Hz

        velocity_cmd = Twist()

        # Main control loop
        while not rospy.is_shutdown():
            # (goal_height + 3 - 0.1)
            if current_height < (z_init - 0.1) and MODEL_MODE == False:  # If below the target height (account for some margin)
                #rospy.loginfo("Ascending... Current height: {:.2f} m".format(current_height))
                velocity_cmd.linear.z = 1.0  # Ascend at 1.0 m/s
                # Publish the velocity command
                vel_publisher.publish(velocity_cmd)
            if current_height >= z_init:
                MODEL_MODE = True
            if MODEL_MODE :
                # start = time.time()
                model_prediction()
                #rospy.loginfo("Hovering at target height: {:.2f} m".format(current_height))
                vel_publisher.publish(model_vel)
                # end = time.time()
                # inference_time  = end - start
                # csv_writer.writerow([inference_time])
                # csv_file.flush()


            

            # Sleep to maintain the loop rate
            rate.sleep()

    except rospy.ROSInterruptException:
        pass

