import requests

# API URL (make sure your Flask app is running)
url = 'http://127.0.0.1:5000/predict'

# Example input data (adjust based on your model's input features)
data = {
    "features": [0.5, 0.7, 0.2, 0.9, 0.1]  # Replace with actual feature values
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Print the API response
print("Response:", response.json())
