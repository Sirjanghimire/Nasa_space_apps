from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib

matplotlib.use('Agg')  # Required for server environments
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)


# Load your model (from your model.py)
def load_model():
    # Your model loading code here
    model = joblib.load('exoseeker_model.pkl')
    return model


model = load_model()


# Serve the frontend
@app.route('/')
def home():
    return render_template('index.html')  # This serves your HTML file


# API endpoint for single prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Prepare features for prediction
        features = [
            data['koi_period'],
            data['koi_duration'],
            data['koi_depth'],
            data['koi_impact'],
            data['koi_model_snr'],
            data['koi_prad'],
            data['koi_teq'],
            data['koi_steff'],
            data['koi_slogg'],
            data['koi_srad'],
            data['koi_kepmag'],
            data['koi_fpflag_nt']
        ]

        # Make prediction
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        # Create SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values([features])

        # Generate SHAP plot
        plt.figure(figsize=(10, 6))
        feature_names = ['Period', 'Duration', 'Depth', 'Impact', 'SNR', 'Radius',
                         'Temp', 'Stellar Temp', 'Stellar Logg', 'Stellar Rad', 'Magnitude', 'FP Flag']

        shap.summary_plot(shap_values, features=[features], feature_names=feature_names, show=False)

        # Convert plot to base64 for frontend
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        shap_plot = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()

        # Prepare response
        class_mapping = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
        prediction_label = class_mapping[prediction]

        response = {
            'prediction': prediction_label,
            'confidence': f"{max(probabilities) * 100:.2f}%",
            'class_probabilities': {
                'FALSE POSITIVE': f"{probabilities[0] * 100:.2f}%",
                'CANDIDATE': f"{probabilities[1] * 100:.2f}%",
                'CONFIRMED': f"{probabilities[2] * 100:.2f}%"
            },
            'shap_plot': shap_plot,
            'analysis_notes': generate_analysis_notes(prediction_label, features)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API endpoint for batch prediction
@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read and process CSV
        df = pd.read_csv(file)

        # Your batch processing logic here
        predictions = []
        for index, row in df.iterrows():
            features = row.tolist()  # Adjust based on your CSV structure
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]

            predictions.append({
                'id': index,
                'prediction': prediction,
                'confidence': max(probabilities)
            })

        # Generate summary
        summary = {
            'total_samples': len(predictions),
            'average_confidence': np.mean([p['confidence'] for p in predictions]),
            'prediction_counts': {
                'CONFIRMED': len([p for p in predictions if p['prediction'] == 2]),
                'CANDIDATE': len([p for p in predictions if p['prediction'] == 1]),
                'FALSE POSITIVE': len([p for p in predictions if p['prediction'] == 0])
            }
        }

        # Create results CSV for download
        results_df = pd.DataFrame(predictions)
        results_csv = results_df.to_csv(index=False)

        response = {
            'summary': summary,
            'results_csv': results_csv
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_analysis_notes(prediction, features):
    notes = []

    if prediction == 'CONFIRMED':
        notes.append("High probability of being a confirmed exoplanet")
        notes.append("Strong transit signal detected")
        notes.append("Stellar parameters within habitable zone range")
    elif prediction == 'CANDIDATE':
        notes.append("Promising exoplanet candidate")
        notes.append("Moderate confidence level")
        notes.append("Further observation recommended")
    else:
        notes.append("Likely false positive detection")
        notes.append("Signal characteristics inconsistent with transits")
        notes.append("Recommend additional verification")

    return notes


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)