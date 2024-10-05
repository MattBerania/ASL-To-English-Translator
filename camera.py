import cv2
import mediapipe as mp
import numpy as np
import requests  # For sending data to Flask

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Import drawing utilities

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Draw landmarks on the frame if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmark_data = []
            for lm in hand_landmarks.landmark:
                landmark_data.append([lm.x, lm.y, lm.z])  # x, y, z coordinates

            # Prepare data for the Flask prediction route
            input_data = np.array(landmark_data).flatten().tolist()  # Flattening and converting to list

            # Send the data to the Flask API for prediction
            try:
                response = requests.post('http://127.0.0.1:5000/predict', json={'landmarks': input_data})
                prediction = response.json().get('letter')  # Updated to match your Flask response
                print(f"Prediction: {prediction}")
            except Exception as e:
                print(f"Error sending landmarks: {e}")

    # Show the frame with OpenCV
    cv2.imshow("ASL Recognition", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
