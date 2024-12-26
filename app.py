from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the scaler, PCA transformer, label encoders, and models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')  # Load the PCA transformer
day_encoder = joblib.load('day_encoder.pkl')
traffic_encoder = joblib.load('traffic_situation_encoder.pkl')  # For Traffic Situation

models = {
    'Linear Regression': joblib.load('Linear Regression.pkl'),
    'K-Nearest Neighbors': joblib.load('K-Nearest Neighbors.pkl'),
    'Decision Tree': joblib.load('Decision Tree.pkl'),
    'Random Forest': joblib.load('Random Forest.pkl'),
    'XGBoost Regressor': joblib.load('XGBoost Regressor.pkl'),
    'Ridge Regression': joblib.load('Ridge Regression.pkl'),
    'Lasso Regression': joblib.load('Lasso Regression.pkl'),
    'Support Vector Machine': joblib.load('Support Vector Machine.pkl'),
}

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs
        model_name = request.form.get('model')
        date = int(request.form.get('Date'))
        day_of_week = request.form.get('Day of the week')
        car_count = float(request.form.get('CarCount'))
        bike_count = float(request.form.get('BikeCount'))
        bus_count = float(request.form.get('BusCount'))
        truck_count = float(request.form.get('TruckCount'))
        total = float(request.form.get('Total'))
        hour = float(request.form.get('Hour'))
        is_weekend = float(request.form.get('IsWeekend'))
        is_peak_hour = float(request.form.get('IsPeakHour'))

        # Encode 'Day of the week' using the same encoder used during training
        if day_of_week not in day_encoder.classes_:
            return render_template('index.html', prediction_text=f"Error: '{day_of_week}' is not a valid day of the week.")
        day_of_week_encoded = day_encoder.transform([day_of_week])[0]

        # Combine all features
        features = [date, day_of_week_encoded, car_count, bike_count, bus_count, truck_count, total, hour, is_weekend, is_peak_hour]

        # Scale the input features
        scaled_features = scaler.transform([features])

        # Apply PCA transformation
        pca_features = pca.transform(scaled_features)

        # Load the selected model
        model = models[model_name]

        # Make prediction
        prediction = model.predict(pca_features)[0]

        # Round prediction for regression models
        if model_name != 'Support Vector Machine':
            prediction = int(np.round(prediction))

        # Convert prediction back to the original class label
        prediction_label = traffic_encoder.inverse_transform([prediction])[0]

        return render_template('index.html', prediction_text=f'Traffic Situation: {prediction_label}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
