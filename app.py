from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib  # For model and scaler deserialization

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler from the .joblib file
model, scaler = joblib.load('trained_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        # Extract features from the received JSON
        features = np.array([data['features']])
        # Standardize the features using the loaded scaler
        features = scaler.transform(features)
        # Make prediction
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    
# Latest Usage:
# curl -X POST -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 1]}" http://54.160.121.180:5000/predict
    
#     usage:
#     CMD:    curl -X POST -H "Content-Type: application/json" -d "{\"features\": [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 1]}" http://localhost:5000/predict

# Output
#features
# {"prediction":0}
# 
# 
# docker run -it -p 5000:5000 9ac991c35151812a719bfbce24f86fd97269f376a8a6dd2dcdfdecef65d2856a
