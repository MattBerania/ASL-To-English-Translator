import cv2
import mediapipe as mp
import numpy as np
import requests  # for sending data to flask

# initializing mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  

# initialize video camera
cap = cv2.VideoCapture(0)

# while the camera is open, read input frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the camera.")
        break

    # mirrors camera
    frame = cv2.flip(frame, 1)

    # converts the frames to RGB since mediapipe and tensorflow are default RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # shows landmarks on hands if detected on camera
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # draws landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extracts landmark data
            landmark_data = []
            for lm in hand_landmarks.landmark:
                landmark_data.append([lm.x, lm.y, lm.z])  # x, y, z coordinates

            # flattens input data and converts to list for prediction
            input_data = np.array(landmark_data).flatten().tolist() 

            # sends the data to flask app for prediction
            try:
                response = requests.post('http://127.0.0.1:5000/predict', json={'landmarks': input_data})
                if response.status_code == 200:
                    prediction = response.json().get('letter')  # get predicted letter from the response
                    print(f"Prediction: {prediction}")
                else:
                    print("Error with API response.")
            except Exception as e:
                print(f"Error sending landmarks: {e}")

    # shows frames using opencv camera
    cv2.imshow("ASL Recognition", frame)

    # exits application when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
