from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.abspath("../ml_model/phishing_detector.pkl")
print("Loading model from:", model_path)  # Debugging line
model = joblib.load(model_path)

# Function to extract features from a URL
def extract_features_from_url(url):
    return {
        "url_length": len(url),
        "num_dots": url.count('.'),
        "has_https": 1 if "https" in url else 0
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Extract features
    features = extract_features_from_url(url)
    features_df = pd.DataFrame([features])

    # Make prediction
    prediction = model.predict(features_df)[0]
    result = "Phishing" if prediction == 1 else "Safe"

    return jsonify({"url": url, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
