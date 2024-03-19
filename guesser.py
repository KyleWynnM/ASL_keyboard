import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import csv

# Load the trained SVM classifier model
model = load("full_alphabet_ratio_based_svm_hand_gesture_classifier.joblib")


# Define a function to calculate ratios between landmark coordinates
def calculate_scaled_ratios(landmarks):
    # Find the lowest and highest landmarks in X, Y, and Z dimensions
    min_x = min(landmarks, key=lambda landmark: landmark.x).x
    max_x = max(landmarks, key=lambda landmark: landmark.x).x
    min_y = min(landmarks, key=lambda landmark: landmark.y).y
    max_y = max(landmarks, key=lambda landmark: landmark.y).y
    min_z = min(landmarks, key=lambda landmark: landmark.z).z
    max_z = max(landmarks, key=lambda landmark: landmark.z).z

    # Calculate scaling factors for X, Y, and Z dimensions
    scale_x = max_x - min_x
    scale_y = max_y - min_y
    scale_z = max_z - min_z

    ratios = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            # Calculate the scaled ratios between landmark coordinates
            ratio_x = (landmarks[i].x - min_x) / scale_x
            ratio_y = (landmarks[i].y - min_y) / scale_y
            ratio_z = (landmarks[i].z - min_z) / scale_z
            ratios.extend([ratio_x, ratio_y, ratio_z])
    return ratios


# Define a function to preprocess hand landmarks data and predict the gesture
def predict_gesture(landmarks):
    # Convert landmark coordinates into ratios
    ratios = calculate_scaled_ratios(landmarks)

    # Reshape the data into a single row
    features = np.array(ratios).reshape(1, -1)

    # Predict the gesture using the trained model
    gesture = model.predict(features)

    return gesture[0]


# initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Start camera loop
while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hand Landmarker
    results = hands.process(rgb_frame)

    # Check if landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect all hand landmarks
            all_landmarks = hand_landmarks.landmark
            # Predict the gesture
            gesture = predict_gesture(all_landmarks)
            # Display the recognized gesture on the screen
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam connection and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
