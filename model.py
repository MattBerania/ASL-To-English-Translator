import mediapipe as mp
import cv2
import numpy as np
import os

from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to extract landmarks
def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return landmarks
    return None

# Example: Extract landmarks from an ASL dataset folder
asl_dataset_path = os.path.expanduser('~/Desktop/ASL-To-English-Translator/asl_alphabet_train')
print(f"ASL dataset path: {asl_dataset_path}")

landmarks_data = []
# labels = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']
labels = []

# Loop over each image in the dataset
for letter_folder in sorted(os.listdir(asl_dataset_path)):
    letter_path = os.path.join(asl_dataset_path, letter_folder)
    if os.path.isdir(letter_path):
        print(f"Processing folder: {letter_folder}")
        for image_file in os.listdir(letter_path):
            image_path = os.path.join(letter_path, image_file)
            print(f"Processing image: {image_file}")
            landmarks = extract_hand_landmarks(image_path)
            if landmarks:
                landmarks_data.append(landmarks)
                labels.append(letter_folder)

def preprocess_landmarks(landmarks_data):
    # Flatten each set of landmarks (21 points, each with x, y, z) into a single 1D array
    flattened_landmarks = [np.array(landmarks).flatten() for landmarks in landmarks_data]
    return np.array(flattened_landmarks)

X_train = preprocess_landmarks(landmarks_data)
y_train = np.array(labels)  # Convert labels to a numpy array


# Convert labels to numerical format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_categorical = to_categorical(y_train_encoded)  # One-hot encoding

# Define a simple neural network model
model = models.Sequential([
    layers.Input(shape=(63,)),  # 21 landmarks with x, y, z (21 * 3 = 63)
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax'),  # Adjust output classes to match the number of unique labels
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("TEST",landmarks_data[:5])  # Print the first 5 extracted landmarks for verification

# Train the model
model.fit(X_train, y_train_categorical, epochs=2, batch_size=128, validation_split=0.2)

def draw_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    for landmark in landmarks:
        # Scale the landmark coordinates to the image dimensions
        h, w, _ = image.shape
        x, y, z = int(landmark[0] * w), int(landmark[1] * h), landmark[2]
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw the landmark
    cv2.imshow('Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage after extracting landmarks
if landmarks:
    draw_landmarks(image_path, landmarks)
