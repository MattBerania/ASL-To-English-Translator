import os
import numpy as np
from tensorflow.keras import Sequential # import this model because we are doing image processing
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# loading preprocessed data
x_train = np.load('x_train.npy')  # loading preprocessed feature data
y_train = np.load('y_train.npy')  # Load preprocessed labels

# define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),  
    Dropout(0.3),  
    Dense(64, activation='relu'),  
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  
])

# compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
history = model.fit(
    x_train,
    y_train,  
    batch_size=32, 
    epochs=20,  # number of times model is trained
    validation_split=0.2  # 20% for validation
)

# evaluate the model
if os.path.exists('x_test.npy') and os.path.exists('y_test.npy'):
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# save the model
model.save('asl_recognition_model.keras')
print("Model saved as 'asl_recognition_model.keras'")

# use the model for predictions
sample = x_train[0]  
sample = np.expand_dims(sample, axis=0)  
prediction = model.predict(sample)
predicted_class = np.argmax(prediction, axis=1)  
print(f"Predicted ASL letter: {predicted_class[0]}")


