import numpy as np
import joblib

# Load the trained model and scaler from the .joblib file
model, scaler = joblib.load('DL_model.joblib')

# Define multiple sets of input data for testing
# Example inputs corresponding to the features the model expects
test_inputs = [
    [7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4, 0],  # Sample for red wine
    [6.7, 0.22, 0.37, 1.4, 0.034, 19, 98, 0.9901, 3.31, 0.46, 11.5, 1],  # Sample for white wine
    [7.3, 0.65, 0, 1.2, 0.065, 15, 21, 0.9946, 3.39, 0.47, 10.0, 0],  # Another red wine sample
    [5.8, 0.29, 0.21, 1.6, 0.044, 32, 88, 0.9932, 3.33, 0.53, 9.7, 1]  # Another white wine sample
]

# Standardize the features using the loaded scaler
test_inputs_scaled = scaler.transform(test_inputs)

# Make predictions
predictions = model.predict(test_inputs_scaled)

# Print out each input and its corresponding prediction
for idx, input_features in enumerate(test_inputs):
    print(f"Input Features: {input_features}")
    print(f"Predicted Quality: {predictions[idx]}")
    print("-" * 60)
