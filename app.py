from flask import Flask, request, jsonify
from model import predict_sign  # imports predict_sign function from model.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")  # confirmation of request reception
    data = request.json  # getting data from the request

    # error handling
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400 
    landmarks = data.get('landmarks')
    if landmarks is None:
        return jsonify({'error': 'No landmarks provided'}), 400  

    # predict the sign letter
    predicted_letter = predict_sign(landmarks)

    return jsonify({'letter': predicted_letter})


if __name__ == '__main__':
    app.run(debug=True)
