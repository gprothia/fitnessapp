import cv2
import mediapipe as mp
import numpy as np
import time

#Adding one line for testing

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize counting variables
counter = 0
last_rep_time = time.time()
msg = ""

# Array to store elbow angles
elbow_angles = []

# Thresholds
WRIST_TOUCH_DISTANCE = 0.2  # Maximum distance between wrists for "touching" (normalized)
RAISED_ELBOW_ANGLE = 140     # Minimum elbow angle to consider arms raised
FORM_DELAY = 1.5             # Minimum time between repetitions in seconds
ANGLE_DIFFERENCE_THRESHOLD = 50  # Minimum difference between max and min elbow angles for good form

def calculate_angle(a, b, c):
    # Calculate angle between three points
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def calculate_distance(a, b):
    # Calculate the Euclidean distance between two points
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Open video feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract coordinates for shoulders, elbows, and wrists
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles for both arms
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Calculate wrist distance
        wrist_distance = calculate_distance(left_wrist, right_wrist)

        # Record elbow angles
        avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
        elbow_angles.append(avg_elbow_angle)

        # Increment counter if arms are raised and wrists are close
        if left_elbow_angle > RAISED_ELBOW_ANGLE and right_elbow_angle > RAISED_ELBOW_ANGLE and wrist_distance < WRIST_TOUCH_DISTANCE:
            if time.time() - last_rep_time > FORM_DELAY:
                counter += 1
                last_rep_time = time.time()

                # Evaluate form
                max_angle = max(elbow_angles)
                min_angle = min(elbow_angles)
                angle_difference = max_angle - min_angle

                if angle_difference >= ANGLE_DIFFERENCE_THRESHOLD:
                    msg = f"Counter {counter} is OK - Good form!"
                else:
                    msg = f"Counter {counter} is NOT OK - Insufficient angle difference."

                # Reset the array for the next repetition
                elbow_angles = []

        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display information
        cv2.putText(frame, f"Reps: {counter}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, msg, (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(frame, f"L Elbow Angle: {int(left_elbow_angle)}", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"R Elbow Angle: {int(right_elbow_angle)}", (50, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Wrist Distance: {wrist_distance:.3f}", (50, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the video feed
    cv2.imshow('Repetition Counter', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
