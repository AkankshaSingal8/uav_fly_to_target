#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
import tensorflow as tf
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras.models import Functional
from keras_models import generate_ncp_model
import threading

# Default camera parameters
IMAGE_SHAPE = (144, 256, 3)  # (Height, Width, Channels)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])  # OpenCV expects (Width, Height)
goal_height = 4

# Load Model
DEFAULT_NCP_SEED = 22222
batch_size = None
seq_len = 64
augmentation_params = None
single_step = True
no_norm_layer = False

mymodel = generate_ncp_model(seq_len, IMAGE_SHAPE, augmentation_params, batch_size, DEFAULT_NCP_SEED, single_step, no_norm_layer)

# Load custom model weights
mymodel.load_weights('retrain_difftraj_wscheduler0.85_seed22222_lr0.001_trainloss0.00016_valloss0.13141_diffcoreset900.h5')

# Extract convolution layers
conv_layers = [layer for layer in mymodel.layers if isinstance(layer, Conv2D)]
act_model_inputs = mymodel.inputs[0]  # Don't want to take in hidden state, just image
vis_model = keras.models.Model(inputs=act_model_inputs, outputs=[layer.output for layer in conv_layers])

# Load goal image safely
goal_image_file = f'./marker_goal_images/goal_marker_height{goal_height}.png'
goal_image = None

if os.path.exists(goal_image_file):
    goal_image = cv2.imread(goal_image_file)
    if goal_image is not None:
        goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    else:
        print(f"Error: Could not read {goal_image_file}")
else:
    print(f"Error: File {goal_image_file} not found.")

SAL_MAP = None
bridge = CvBridge()
image_lock = threading.Lock()


def compute_visualbackprop(img, activation_model):
    """
    Compute saliency map using VisualBackprop.
    """
    kernels, strides = [], []
    
    for layer in activation_model.layers[1:]:
        if isinstance(layer, Conv2D):
            kernels.append(layer.kernel_size)
            strides.append(layer.strides)

    activations = activation_model.predict(img)
    average_layer_maps = []

    for layer_activation in activations:
        feature_maps = layer_activation[0]
        average_feature_map = np.sum(feature_maps, axis=-1) / feature_maps.shape[-1]

        # Normalize
        map_min = np.min(average_feature_map)
        map_max = np.max(average_feature_map)
        normal_map = (average_feature_map - map_min) / (map_max - map_min + 1e-6)

        average_layer_maps.append(normal_map)

    average_layer_maps = [fm[np.newaxis, :, :, np.newaxis] for fm in average_layer_maps]
    saliency_mask = tf.convert_to_tensor(average_layer_maps[-1])

    for l in reversed(range(0, len(average_layer_maps))):
        kernel = np.ones((*kernels[l], 1, 1))

        if l > 0:
            output_shape = average_layer_maps[l - 1].shape
        else:
            output_shape = (1, *(IMAGE_SHAPE[:2]), 1)

        saliency_mask = tf.nn.conv2d_transpose(saliency_mask, kernel, output_shape, strides[l], padding='VALID')
        if l > 0:
            saliency_mask = tf.multiply(saliency_mask, average_layer_maps[l - 1])

    saliency_mask = tf.squeeze(saliency_mask, axis=0)
    return saliency_mask


def plot_saliency_map(saliency_map, image_size):
    """
    Convert saliency map to heatmap.
    """
    saliency_map = (saliency_map - tf.reduce_min(saliency_map)) / (tf.reduce_max(saliency_map) - tf.reduce_min(saliency_map) + 1e-6)
    saliency_map = saliency_map.numpy() if isinstance(saliency_map, tf.Tensor) else saliency_map

    if saliency_map.shape[:2] != image_size:
        saliency_map = tf.image.resize(saliency_map, image_size).numpy()

    saliency_map = np.uint8(255 * saliency_map)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    return saliency_map


def converter(data):
    """
    ROS callback function to process camera feed.
    """
    global SAL_MAP, goal_image

    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        cv_img = cv2.resize(cv_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))

        if goal_image is None:
            print("Error: Goal image is not loaded.")
            return

        # Ensure both images have the same shape and type
        goal_image_resized = cv2.resize(goal_image, (cv_img.shape[1], cv_img.shape[0]))
        goal_image_resized = goal_image_resized.astype(np.uint8)
        cv_img = cv_img.astype(np.uint8)

        difference = cv2.absdiff(cv_img, goal_image_resized)

        img_array = np.expand_dims(difference, axis=0)
        img_tensor = tf.convert_to_tensor(img_array)

        saliency_map = compute_visualbackprop(img_tensor, vis_model)
        sal_img = plot_saliency_map(saliency_map, (IMAGE_SHAPE_CV))

        with image_lock:
            SAL_MAP = sal_img

    except CvBridgeError as e:
        print(f"CV Bridge Error: {e}")


def show_sal_map():
    """
    Display the saliency map safely in the main thread.
    """
    global SAL_MAP

    with image_lock:
        if SAL_MAP is not None:
            cv2.imshow("Saliency Map", SAL_MAP)
            cv2.waitKey(1)


def camera_feed():
    """
    Initializes the ROS node and starts the camera feed.
    """
    rospy.init_node('saliency_map_node', anonymous=True)
    rospy.Subscriber("/hector/downward_cam/camera/image", Image, converter)

    rate = rospy.Rate(30)  # 30Hz

    while not rospy.is_shutdown():
        show_sal_map()
        rate.sleep()

    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        camera_feed()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
