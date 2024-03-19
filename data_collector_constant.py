import cv2
import mediapipe as mp
import csv


# Define a function to calculate ratios between landmarks with scaling
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


# Modify the write_ratios_to_csv function to use calculate_scaled_ratios
def write_ratios_to_csv(landmarks, key, csv_file):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [key]
        ratios = calculate_scaled_ratios(landmarks)
        row.extend(ratios)
        writer.writerow(row)

# Initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define the path to save the CSV file
csv_file = "hand_landmarks_data.csv"

# Variable to store the active letter
active_letter = None

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
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Collect all hand landmarks and write ratios to CSV for the active letter
        if active_letter is not None:
            all_landmarks = hand_landmarks.landmark
            write_ratios_to_csv(all_landmarks, active_letter, csv_file)

    # Display the active letter on the frame
    if active_letter:
        cv2.putText(frame, f'Active Letter: {active_letter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('ASL Recognition', frame)

    # Check for key press to set active letter
    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # Some key is pressed
        if key == ord('.'):  # If '.' is pressed, set active letter to None
            active_letter = None
        else:
            active_letter = chr(key)  # Convert key code to character

            # Check if the user wants to stop ('`' key)
            if active_letter == '`':
                break

# Release the webcam connection and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
