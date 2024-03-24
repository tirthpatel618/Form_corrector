import numpy as np

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Mid point (e.g., elbow)
    c = np.array(c)  # End point (e.g., wrist)

    # Calculate the angle in radians and then convert it to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Ensure the angle is less than 180 degrees
    if angle > 180.0:
        angle = 360 - angle
    
    return angle