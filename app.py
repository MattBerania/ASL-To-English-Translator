from flask import Flask, request, jsonify
from model import predict_sign

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")  # Add this line
    data = request.json  # Get data from the request
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400  # Add this line
    landmarks = data.get('landmarks')
    if landmarks is None:
        return jsonify({'error': 'No landmarks provided'}), 400
    predicted_letter = predict_sign(landmarks)
    return jsonify({'letter': predicted_letter})


if __name__ == '__main__':
    app.run(debug=True)
