
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load Crop Recommendation model
crop_model = joblib.load("crop_model.pkl")
crop_scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Load Crop Yield Prediction model
yield_data = joblib.load("crop_yield_prediction.joblib")

yield_model = yield_data["model"]
yield_scaler = yield_data["scaler"]
yield_features = yield_data["features"]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/crop')
def crop():
    return render_template("crop.html")


@app.route('/yield')
def yield_page():
    return render_template("yield.html")


# CROP RECOMMENDATION
@app.route('/predict_crop', methods=['POST'])
def predict_crop():

    try:
        data = pd.DataFrame([[
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]], columns=[
            'N', 'P', 'K', 'temperature',
            'humidity', 'ph', 'rainfall'
        ])

    except (ValueError, KeyError):
        return render_template(
            "crop.html",
            error="Please enter valid numbers in all fields."
        )

    # Check for unrealistic input values
    limits = {
        'N': (0, 200),
        'P': (0, 200),
        'K': (0, 250),
        'temperature': (0, 60),
        'humidity': (0, 100),
        'ph': (0, 14),
        'rainfall': (0, 5000)
    }

    for column, (minimum, maximum) in limits.items():
        value = data[column].iloc[0]

        if not minimum <= value <= maximum:
            return render_template(
                "crop.html",
                error=f"{column} must be between {minimum} and {maximum}."
            )

    # Scale input and predict
    data_scaled = crop_scaler.transform(data)
    prediction = crop_model.predict(data_scaled)
    crop_name = le.inverse_transform(prediction)[0]

    return render_template(
        "crop.html",
        prediction=crop_name
    )


# CROP YIELD PREDICTION
@app.route('/predict_yield', methods=['POST'])
def predict_yield():

    try:
        crop_name = request.form['crop'].strip().lower()
        state = request.form['state'].strip().lower()
        season = request.form['season'].strip().lower()

        # Create DataFrame with the model's training features
        data = pd.DataFrame(
            0, index=[0], columns=yield_features
        )

        # Numeric inputs
        data['Crop_Year'] = float(request.form['crop_year'])
        data['Area'] = float(request.form['area'])
        data['Annual_Rainfall'] = float(request.form['rainfall'])
        data['Fertilizer'] = float(request.form['fertilizer'])
        data['Pesticide'] = float(request.form['pesticide'])
        data['Avg_Temperature'] = float(
            request.form['avg_temperature']
        )

    except (ValueError, KeyError):
        return render_template(
            "yield.html",
            error="Please enter valid numbers in all fields."
        )

    # One-hot encoding for crop, season and state
    crop_found = False
    season_found = False
    state_found = False

    for col in yield_features:

        if col.startswith('Crop_') and col[5:].strip().lower() == crop_name:
            data[col] = 1
            crop_found = True

        elif col.startswith('Season_') and col[7:].strip().lower() == season:
            data[col] = 1
            season_found = True

        elif col.startswith('State_') and col[6:].strip().lower() == state:
            data[col] = 1
            state_found = True

    if not crop_found or not season_found or not state_found:
        return render_template(
            "yield.html",
            error="Crop, state or season not found in the training dataset."
        )

    # Match training feature order
    data = data[yield_features]

    # Apply the same scaler used during training
    data_scaled = yield_scaler.transform(data)

    # Predict yield
    prediction = yield_model.predict(data_scaled)[0]

    return render_template(
        "yield.html",
        prediction=round(float(prediction), 2)
    )


if __name__ == '__main__':
    app.run(debug=True)
