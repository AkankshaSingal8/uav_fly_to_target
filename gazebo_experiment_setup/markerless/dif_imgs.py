import os
import cv2

root = "output_images"
file_ending = 'png'
destination = "diff_images"

image_files = [fn for fn in os.listdir(root) if file_ending in fn]


# Load the goal images
goal_image = cv2.imread('markerless_goal_images/goal_markerless_height3.png')
n_images = len(image_files)

for i in range(1, n_images + 1):
	current_image = cv2.imread(f'{root}/Image{i}.png')

	difference = cv2.absdiff(current_image, goal_image)

	# Save the result
	cv2.imwrite(f'{destination}/Image{i}.png', difference)
    
    
