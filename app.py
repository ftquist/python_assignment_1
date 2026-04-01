from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib, json, os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE, 'model.pkl'))
scaler = joblib.load(os.path.join(BASE, 'scaler.pkl'))
stats  = json.load(open(os.path.join(BASE, 'stats.json')))

# Encoding maps (must match training order)
from sklearn.preprocessing import LabelEncoder
import pickle

# We'll reproduce encoding logic at inference
CAT_MAPS = {
    'fuel_type':      {v:i for i,v in enumerate(sorted(stats['fuel_type']))},
    'gear_box_type':  {v:i for i,v in enumerate(sorted(stats['gear_box_type']))},
    'drive_wheels':   {v:i for i,v in enumerate(sorted(stats['drive_wheels']))},
    'category':       {v:i for i,v in enumerate(sorted(stats['category']))},
}

FEATURE_ORDER = [
    'car_age','log_mileage','log_levy','engine_volume',
    'cylinders','airbags','is_leather','is_left_wheel',
    'fuel_type_enc','gear_box_type_enc','drive_wheels_enc','category_enc'
]

@app.route('/')
def index():
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        production_year = int(d['production_year'])
        mileage         = float(d['mileage'])
        levy            = float(d.get('levy', 0))
        engine_volume   = float(d['engine_volume'])
        cylinders       = float(d['cylinders'])
        airbags         = float(d['airbags'])
        is_leather      = 1 if d.get('leather_interior') == 'Yes' else 0
        is_left_wheel   = 1 if d.get('wheel') == 'Left wheel' else 0
        fuel_type       = d['fuel_type']
        gear_box_type   = d['gear_box_type']
        drive_wheels    = d['drive_wheels']
        category        = d['category']

        car_age     = 2024 - production_year
        log_mileage = np.log1p(mileage)
        log_levy    = np.log1p(levy)

        row = np.array([[
            car_age, log_mileage, log_levy, engine_volume,
            cylinders, airbags, is_leather, is_left_wheel,
            CAT_MAPS['fuel_type'].get(fuel_type, 0),
            CAT_MAPS['gear_box_type'].get(gear_box_type, 0),
            CAT_MAPS['drive_wheels'].get(drive_wheels, 0),
            CAT_MAPS['category'].get(category, 0),
        ]])

        row_scaled = scaler.transform(row)
        price = model.predict(row_scaled)[0]
        price = max(0, round(price, 2))

        return jsonify({'status': 'ok', 'price': price})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    app.run(host='0.0.0.0', port=port)
