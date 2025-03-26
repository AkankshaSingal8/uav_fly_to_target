import cv2
import numpy as np

# Image and recording settings
IMAGE_SHAPE = (144, 256, 3)  # Target image size (height, width, channels)
IMAGE_SHAPE_CV = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])  # OpenCV shape (width, height)

GOAL_IMAGE_FILE = "./markerless_goal_images/goal_markerless_height3.png"
goal_image = cv2.imread(GOAL_IMAGE_FILE)
goal_image = cv2.resize(goal_image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
goal_image = goal_image.astype(np.float32) / 255.0  # Normalize to [0,1]

cv_img = cv2.imread(GOAL_IMAGE_FILE)
cv_img = cv2.resize(cv_img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
cv_img = cv_img.astype(np.float32) / 255.0  # Normalize to [0,1]

# Compute Mean Squared Error (MSE) Loss
mse_loss = np.mean((cv_img - goal_image) ** 2)
diff = (cv_img - goal_image) ** 2
print(diff.shape)
print(mse_loss)