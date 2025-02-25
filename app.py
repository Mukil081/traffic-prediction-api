from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model from the .pkl file
with open('traffic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from request
        features = np.array(data['features']).reshape(1, -1)  # Convert to array
        prediction = model.predict(features)  # Make prediction
        return jsonify({'prediction': int(prediction[0])})  # Return as JSON
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
