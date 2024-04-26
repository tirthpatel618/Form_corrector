import numpy as np
import mediapipe as mp
import numpy as np
import cv2
import math

# Initialize mediapipe drawing and pose components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

def get_angles():
    # Initialize video capture
    elbow_angle = []
    hand_angle = []
    hip_angle = []
    cap = cv2.VideoCapture('/Users/tirthpatel/Desktop/project3/free_throw_made.mp4')
    count = 1
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make a detection
            results = pose.process(image)

            # Draw the pose landmarks on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Get the landmarks
                landmarks = results.pose_landmarks.landmark

                # Get the landmarks for the right arm
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]


                # Calculate the angle
                angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                cv2.putText(image, str(angle), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elbow_angle.append([count, round(angle, 1)]) # Append the angle to the list
                angle = calculate_angle(right_elbow, right_wrist, right_hand)
                hand_angle.append([count, round(angle, 1)])
                angle = calculate_angle(right_hip, right_shoulder, right_elbow)
                hip_angle.append([count,round(angle, 1)])
                count += 1
    cap.release()
    cv2.destroyAllWindows()
    return elbow_angle, hand_angle, hip_angle


#get angles
elbow_angle, hand_angle, hip_angle = get_angles()
print(elbow_angle)
print(hand_angle)
print(hip_angle)