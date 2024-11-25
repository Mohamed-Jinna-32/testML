from flask import Flask, request, jsonify
import numpy as np
import pickle


app = Flask(__name__)


with open('scaler', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('final_logistic_regression_model', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Logistic Regression Model API"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get("data")
        
        if not features or not isinstance(features, list):
            return jsonify({"error": "Invalid input. Provide 'data' as a list of feature values."}), 400
        
        features = np.array(features).reshape(1, -1)
        
        scaled_features = loaded_scaler.transform(features)
        
        prediction = int(loaded_model.predict(scaled_features)[0])
        probability = float(loaded_model.predict_proba(scaled_features)[0, 1])
        
        return jsonify({
            "prediction": prediction,
            "probability": probability
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
