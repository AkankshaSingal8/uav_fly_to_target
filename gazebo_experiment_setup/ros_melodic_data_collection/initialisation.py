#!/usr/bin/env python
import subprocess
import time

# Define the initialization values
initialization = [
    [(-2.0, -2.0, 8.0), (0.0, 0.0, 0.0, 1.0)],
    [(-2.0, -2.0, 8.0), (0.0, 0.0, 1.0, 0.0)]
]

# Path to the main script
script_path = "./data_collection_final.py"  # Replace with the actual path to your script

# Iterate through the initialization values and run the script
for pose in initialization:
    position, orientation = pose
    position_args = [str(coord) for coord in position]
    orientation_args = [str(coord) for coord in orientation]

    # Combine arguments for subprocess
    args = ["python", script_path] + position_args + orientation_args

    print("Running script with position", position ,"and orientation", orientation)
    process = subprocess.Popen(args)

    # Wait for the process to finish (if needed)
    process.wait()

    # Add a delay between iterations (optional)
    time.sleep(5)

