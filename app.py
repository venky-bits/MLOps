from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Ensure input is a list of records
        if isinstance(data, dict):
            data = [data]  # Wrap a single dictionary in a list
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame(data)

        # Validate input data
        if input_data.empty:
            return jsonify({'error': 'Input data is empty'}), 400

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)