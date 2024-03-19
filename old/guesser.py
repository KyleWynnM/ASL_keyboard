import cv2
import mediapipe as mp
import numpy as np
from joblib import load

# Load the trained SVM classifier model
model = load("abcooo_svm_hand_gesture_classifier.joblib")


# Define a function to preprocess hand landmarks data and predict the gesture
def predict_gesture(landmarks):
    # Extract x, y, z coordinates of all landmarks
    landmark_coords = []
    for landmark in landmarks:
        landmark_coords.extend([landmark.x, landmark.y, landmark.z])

    # Reshape the data into a single row
    features = np.array(landmark_coords).reshape(1, -1)

    # Predict the gesture using the trained model
    gesture = model.predict(features)

    return gesture[0]


# initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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
