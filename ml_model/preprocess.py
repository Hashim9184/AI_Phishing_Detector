import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 3: Extracting Features from Dataset
def extract_features(data):
    if data is not None:
        # Example feature extraction (modify based on dataset structure)
        data['url_length'] = data['url'].apply(lambda x: len(x))
        data['num_dots'] = data['url'].apply(lambda x: x.count('.'))
        data['has_https'] = data['url'].apply(lambda x: 1 if 'https' in x else 0)
        print("Features extracted successfully.")
        return data
    else:
        print("No data available for feature extraction.")
        return None

# Step 4: Training the Machine Learning Model
def train_model(data):
    if data is not None:
        features = ['url_length', 'num_dots', 'has_https']
        X = data[features]
        y = data['label']  # Assuming 'label' column exists (1 for phishing, 0 for legitimate)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        ml_model_path = "../ml_model"
        joblib.dump(model, os.path.join(ml_model_path, "phishing_detector.pkl"))
        print("Model saved successfully.")
        return model
    else:
        print("No data available for training.")
        return None

# Example usage
dataset_path = os.path.abspath("..//datasets/phishing_data.csv")
dataset_path = load_dataset(dataset_path)
dataset_path = extract_features(dataset_path)
model = train_model(dataset_path)