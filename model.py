import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

# load the trained model
model = load_model('asl_recognition_model.keras')

# convert and save it to the web folder for client side version
tfjs.converters.save_keras_model(model, 'web')


# load the label encoder so we can decode
from sklearn.preprocessing import LabelEncoder
import pickle

# load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def predict_sign(landmarks):
    """
    Given the landmarks of a hand gesture, predict the corresponding ASL letter.

    Args:
        landmarks (list): List of (x, y, z) landmarks of the hand.

    Returns:
        str: Predicted ASL letter.
    """
    # flatten the landmarks into a 1D array
    flattened_landmarks = np.array(landmarks).flatten().reshape(1, -1)

    # make prediction
    prediction = model.predict(flattened_landmarks)
    
    # get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # map the predicted index to the corresponding ASL letter by decoding 
    predicted_class = label_encoder.inverse_transform([predicted_class_index])

    return predicted_class[0]
