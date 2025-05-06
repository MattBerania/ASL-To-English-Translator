import mediapipe as mp
import cv2
import os
import numpy as np
import pickle

# initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# function to extract landmarks on hand images
def extract_hand_landmarks(image_path):
    # read the image
    image = cv2.imread(image_path)
    
    # convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # process the image to extract landmarks
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        # if landmarks are detected, return them as a list of (x, y, z) coordinates
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return landmarks
    return None


# path to our ASL dataset folder 
asl_dataset_path = './asl_dataset' 

# initialize container lists for the landmark data and labels 
# (which will associate the coordinates to the corresponding letter/number)
landmarks_data = []
labels = []

# loop through the images dataset 
for letter_folder in sorted(os.listdir(asl_dataset_path)):
    letter_path = os.path.join(asl_dataset_path, letter_folder)
    
    if os.path.isdir(letter_path):
        print(f"Processing folder: {letter_folder}")
        
        # loop through each image in the folder
        for image_file in os.listdir(letter_path):
            image_path = os.path.join(letter_path, image_file)
            print(f"Processing image: {image_file}")
            
            # extract landmarks for the image
            landmarks = extract_hand_landmarks(image_path)
            
            if landmarks:
                # append landmarks and corresponding label (folder name corresponds to the letter)
                landmarks_data.append(landmarks)
                labels.append(letter_folder)


# main function
def preprocess_landmarks(landmarks_data):
    # flatten each set of landmarks (21 points with x, y, z each) into a single 1D array
    flattened_landmarks = [np.array(landmarks).flatten() for landmarks in landmarks_data]
    return np.array(flattened_landmarks)

# preprocess the landmarks
x_train = preprocess_landmarks(landmarks_data)
y_train = np.array(labels)  # corresponding labels for each image (ASL letter)


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# convert the results into numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# convert the labels to categorical format (one hot encoding/binary) to train model
y_train_categorical = to_categorical(y_train_encoded)


# save the label encoder to a file with pickle
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

print("Label encoder saved to 'label_encoder.pkl'.")

# drawing the lines on hand
def draw_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    for landmark in landmarks:
        # scale landmark coords to image dimensions
        h, w, _ = image.shape
        x, y, z = int(landmark[0] * w), int(landmark[1] * h), landmark[2]
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  
    
    # show image with landmarks drawn
    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# save preprocessed data to .npy files
np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train_categorical)

print("Data saved to x_train.npy and y_train.npy.")
