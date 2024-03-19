import cv2
import mediapipe as mp
import numpy as np
import csv

# Define a function to write hand landmarks and key to a CSV file
def write_to_csv(landmarks, key, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [key]
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z])  # Assuming x, y, z coordinates are required
        writer.writerow(row)

# initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define the path to save the CSV file
csv_file = "hand_landmarks_data.csv"

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
            # Check for user input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Some key is pressed
                key = chr(key)  # Convert key code to character
                write_to_csv(all_landmarks, key, csv_file)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam connection and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
