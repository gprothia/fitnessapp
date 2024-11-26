import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize counting variables
counter = 0
msg = ""
form_correct = False
angles = []  # Array to store angles within each repetition
increasing_streak = 0  # Counter to track consecutive increasing angles
angle_history = deque(maxlen=5)  # Smoothing window for moving average
last_rep_time = time.time()  # To track the time between repetitions

# Thresholds for detecting a repetition
MIN_MOVEMENT_THRESHOLD = 30    # Minimum angle to detect the lowest point in curl
MAX_MOVEMENT_THRESHOLD = 150  # Angle threshold to detect full extension
MIN_RANGE_OF_MOTION = 100      # Minimum range of motion to validate a repetition
REPETITION_DELAY = 1.5         # Delay in seconds between repetitions to prevent double-counting

def moving_average(data):
    return sum(data) / len(data)

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

# Open video feed
cap = cv2.VideoCapture(0)

# State tracking for repetition
movement_started = False
stable_low = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and process with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Extract landmarks for shoulder, elbow, and wrist
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angle and append to smoothing window
        angle = calculate_angle(shoulder, elbow, wrist)
        angle_history.append(angle)
        
        # Apply smoothing
        smooth_angle = moving_average(angle_history)
        angles.append(smooth_angle)

        # Track the lowest point (start of curl)
        if smooth_angle < MIN_MOVEMENT_THRESHOLD:
            stable_low = True  # Reached lowest point, stable enough to start curl

        if stable_low  and (time.time() - last_rep_time > REPETITION_DELAY):
            max_angle = max(angles)
            min_angle = min(angles)
            range_of_motion = max_angle - min_angle

            # Check if range of motion and angle thresholds are met for a good repetition
            if range_of_motion > MIN_RANGE_OF_MOTION and smooth_angle > MAX_MOVEMENT_THRESHOLD:
                if max_angle > MAX_MOVEMENT_THRESHOLD and min_angle < MIN_MOVEMENT_THRESHOLD:
                    counter += 1
                    form_correct = True
                    msg = f"Repetition {counter}: Good form!"
                    print(msg)
            elif max_angle <= MAX_MOVEMENT_THRESHOLD:
                counter += 1
                form_correct = False
                msg = f"Repetition {counter}: Not good form - Arm needs to be stretched more."
                print(msg)
            elif min_angle >= MIN_MOVEMENT_THRESHOLD:
                counter += 1
                form_correct = False
                msg = f"Repetition {counter}: Not good form - Arm needs to be curled more ."
                print(msg)
        
            # Reset for the next repetition
            angles = []
            last_rep_time = time.time()
            stable_low = False  # Reset stable low for next repetition
        


        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display angle, counter, and form message for debugging
        cv2.putText(frame, f"Angle: {int(smooth_angle)}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Reps: {counter}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Form: {msg}", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow('Bicep Curl Counter', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
