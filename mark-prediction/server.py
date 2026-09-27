import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

app = Flask(__name__, static_folder='static')
CORS(app)

DATA_PATH = os.path.join("data", "student_scores.csv")
MODEL_PATH = os.path.join("models", "student_marks_model.joblib")
FEATURES = ['Hours Studied', 'Sleep Hours', 'Attendance Percentage', 'Previous Marks']
TARGET = 'Marks Scored'

def get_model_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Simple Linear Regression
    simple_model = LinearRegression()
    simple_model.fit(X_train[['Hours Studied']], y_train)
    y_pred_simple = simple_model.predict(X_test[['Hours Studied']])
    r2_simple = float(r2_score(y_test, y_pred_simple))
    mae_simple = float(mean_absolute_error(y_test, y_pred_simple))
    rmse_simple = float(np.sqrt(mean_squared_error(y_test, y_pred_simple)))

    # Model 2: Multiple Linear Regression
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred_multi = multi_model.predict(X_test)
    r2_multi = float(r2_score(y_test, y_pred_multi))
    mae_multi = float(mean_absolute_error(y_test, y_pred_multi))
    mse_multi = float(mean_squared_error(y_test, y_pred_multi))
    rmse_multi = float(np.sqrt(mse_multi))

    # Feature Importance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaled_model = LinearRegression()
    scaled_model.fit(X_train_scaled, y_train)
    
    beta_coefs = scaled_model.coef_
    abs_beta = np.abs(beta_coefs)
    total_beta = np.sum(abs_beta)
    impact_pct = (abs_beta / total_beta) * 100
    
    raw_coefs = dict(zip(FEATURES, [float(c) for c in multi_model.coef_]))
    impact_dict = [
        {
            "feature": feat,
            "raw_coef": round(float(multi_model.coef_[i]), 4),
            "std_beta": round(float(beta_coefs[i]), 4),
            "impact_pct": round(float(impact_pct[i]), 1)
        }
        for i, feat in enumerate(FEATURES)
    ]
    impact_dict.sort(key=lambda x: x['impact_pct'], reverse=True)

    os.makedirs("models", exist_ok=True)
    joblib.dump(multi_model, MODEL_PATH)

    return {
        "features": FEATURES,
        "raw_coefs": raw_coefs,
        "intercept": round(float(multi_model.intercept_), 4),
        "metrics": {
            "multi": {"r2": round(r2_multi, 4), "mae": round(mae_multi, 2), "rmse": round(rmse_multi, 2), "mse": round(mse_multi, 2)},
            "simple": {"r2": round(r2_simple, 4), "mae": round(mae_simple, 2), "rmse": round(rmse_simple, 2)}
        },
        "feature_impact": impact_dict,
        "top_feature": impact_dict[0]["feature"],
        "samples": len(df)
    }

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/info', methods=['GET'])
def get_info():
    info = get_model_data()
    if not info:
        return jsonify({"error": "Dataset not found"}), 404
        
    df = pd.read_csv(DATA_PATH)
    info['dataset'] = df.to_dict(orient='records')
    info['null_counts'] = df.isnull().sum().to_dict()
    info['summary'] = df.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_dict(orient='index')
    return jsonify(info)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        h = float(data.get('hours', 6.5))
        s = float(data.get('sleep', 7.0))
        a = float(data.get('attendance', 80.0))
        p = float(data.get('previous_marks', 70.0))
        
        if not os.path.exists(MODEL_PATH):
            get_model_data()
            
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([[h, s, a, p]], columns=FEATURES)
        raw_pred = float(model.predict(input_df)[0])
        clamped_score = max(0.0, min(100.0, raw_pred))

        if clamped_score >= 90: grade = "A+ (Outstanding)"
        elif clamped_score >= 80: grade = "A (Excellent)"
        elif clamped_score >= 70: grade = "B (Good)"
        elif clamped_score >= 60: grade = "C (Satisfactory)"
        elif clamped_score >= 50: grade = "D (Pass)"
        else: grade = "F (Needs Improvement)"

        return jsonify({
            "inputs": {"hours": h, "sleep": s, "attendance": a, "previous_marks": p},
            "predicted_score": round(clamped_score, 2),
            "raw_prediction": round(raw_pred, 2),
            "grade": grade
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    get_model_data()
    print("Starting Flask API server on http://localhost:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
