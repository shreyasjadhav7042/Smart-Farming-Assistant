from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


# Crop Recommendation Model
crop_model = joblib.load('crop_model.pkl')
crop_scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')


# Yield Prediction Model
yield_data = joblib.load('crop_yield_model.joblib')

yield_model = yield_data['model']
yield_scaler = yield_data['scaler']
yield_features = yield_data['features']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/crop')
def crop():
    return render_template('crop.html')


@app.route('/yield')
def yield_page():
    return render_template('yield.html')


@app.route('/predict_crop', methods=['POST'])
def predict_crop():

    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    data = pd.DataFrame([[
        N, P, K, temperature,
        humidity, ph, rainfall
    ]], columns=[
        'N', 'P', 'K', 'temperature',
        'humidity', 'ph', 'rainfall'
    ])

    data_scaled = crop_scaler.transform(data)

    prediction = crop_model.predict(data_scaled)

    crop_name = le.inverse_transform(prediction)[0]

    return render_template(
        'crop.html',
        prediction=crop_name
    )


@app.route('/predict_yield', methods=['POST'])
def predict_yield():

    crop = request.form['crop']
    state = request.form['state']
    season = request.form['season']

    crop_year = float(request.form['crop_year'])
    area = float(request.form['area'])
    rainfall = float(request.form['rainfall'])
    fertilizer = float(request.form['fertilizer'])
    pesticide = float(request.form['pesticide'])
    avg_temperature = float(request.form['avg_temperature'])
    max_temperature = float(request.form['max_temperature'])
    min_temperature = float(request.form['min_temperature'])


    # Create dataframe with all required features
    data = pd.DataFrame(0, index=[0], columns=yield_features)


    # Numeric features
    data['Crop_Year'] = crop_year
    data['Area'] = area
    data['Annual_Rainfall'] = rainfall
    data['Fertilizer'] = fertilizer
    data['Pesticide'] = pesticide
    data['Avg_Temperature'] = avg_temperature
    data['Max_Temperature'] = max_temperature
    data['Min_Temperature'] = min_temperature


    # Find matching one-hot columns
    crop_column = next(
        col for col in yield_features
        if col.startswith('Crop_') and col[5:].strip().lower() == crop.strip().lower()
    )

    season_column = next(
        col for col in yield_features
        if col.startswith('Season_') and col[7:].strip().lower() == season.strip().lower()
    )

    state_column = next(
        col for col in yield_features
        if col.startswith('State_') and col[6:].strip().lower() == state.strip().lower()
    )


    # Set selected categories to 1
    data[crop_column] = 1
    data[season_column] = 1
    data[state_column] = 1


    # Scale
    data_scaled = yield_scaler.transform(data)


    # Predict
    prediction = yield_model.predict(data_scaled)[0]


    return render_template(
        'yield.html',
        prediction=round(prediction, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)