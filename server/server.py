from flask import Flask, jsonify
from flask_cors import CORS
# app instance

app = Flask(__name__)
CORS(app)

# only route set up is api/home
@app.route("/api/home", methods=['GET'])
def return_home():
    return jsonify({
        'message': "Hello World!"
    })
if __name__ == "__main__":
    app.run(debug=True, port=8080)