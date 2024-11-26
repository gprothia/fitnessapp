import cv2
import mediapipe as mp
import numpy as np
import time

import os

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Initialize counting variables
counter = 0
direction = None  # Tracks the movement direction: "up" or "down"
angle_history = []  # Array to store wrist positions (y-coordinates)
WINDOW_SIZE = 5  # Moving average window size
THRESHOLD_PERCENTAGE = 0.02  # Minimum 2% change for direction detection

def moving_average(data, window_size):
    """Calculate the moving average of the last `window_size` elements"""
    if len(data) < window_size:
        return None
    return np.mean(data[-window_size:])


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

        # Extract y-coordinates for wrists
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y

        # Average wrist positions for a single y-coordinate
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        angle_history.append(avg_wrist_y)

        # Initialize averages for display
        current_avg = None
        previous_avg = None

        # Ensure enough data points for moving average comparison
        if len(angle_history) >= WINDOW_SIZE * 2:
            # Calculate moving averages
            current_avg = moving_average(angle_history, WINDOW_SIZE)
            previous_avg = moving_average(angle_history[-WINDOW_SIZE * 2:-WINDOW_SIZE], WINDOW_SIZE)

            # Check for directional change with 2% threshold
            if current_avg and previous_avg:
                percentage_change = abs((current_avg - previous_avg) / previous_avg)
                if percentage_change >= THRESHOLD_PERCENTAGE:
                    if current_avg < previous_avg and direction != "up":  # Movement going up
                        counter += 1
                        direction = "up"
                        #announce_repetition(counter, direction)
                    elif current_avg > previous_avg and direction != "down":  # Movement going down
                        #counter += 1
                        direction = "down"
                        #announce_repetition(counter, direction)

        # Draw landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display information
        cv2.putText(frame, f"Reps: {counter}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {direction}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Curr Avg Y: {current_avg if current_avg else 0:.3f}", (50, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Prev Avg Y: {previous_avg if previous_avg else 0:.3f}", (50, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Display the video feed
    cv2.imshow('Dumbbell Shoulder Fly Counter', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
