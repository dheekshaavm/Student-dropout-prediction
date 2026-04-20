from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

from shap_utils import explain_prediction

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("model/model.pkl")

# Load training feature columns once (IMPORTANT)
train_df = pd.read_csv("model/processed_data.csv")
feature_columns = train_df.drop("risk", axis=1).columns


# ==============================
# 🧠 INTERVENTION ENGINE
# ==============================
def generate_interventions(df):
    interventions = []

    # Absence-based suggestions
    if df.get("absences", 0).iloc[0] > 10:
        interventions.append("Monitor attendance weekly")

    # Academic weakness
    if df.get("G1", 0).iloc[0] < 10 or df.get("G2", 0).iloc[0] < 10:
        interventions.append("Provide academic support")

    # Failure history
    if df.get("failures", 0).iloc[0] > 0:
        interventions.append("Assign mentor for subject guidance")

    # Low study time
    if df.get("studytime", 0).iloc[0] <= 1:
        interventions.append("Encourage structured study schedule")

    # Internet access issues
    if df.get("internet", 1).iloc[0] == 0:
        interventions.append("Provide access to digital learning resources")

    # Default suggestion if nothing triggered
    if not interventions:
        interventions.append("Maintain regular performance monitoring")

    return interventions


# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return {"message": "Student Dropout AI API running 🚀"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Add missing columns with default value 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct column order
        df = df[feature_columns]

        # Make prediction
        prob = model.predict_proba(df)[0][1]
        pred = int(prob > 0.5)

        # SHAP explanation
        shap_vals = explain_prediction(df)[0].tolist()

        # 🧠 Generate AI interventions
        interventions = generate_interventions(df)

        return jsonify({
            "prediction": pred,
            "probability": float(prob),
            "shap_values": shap_vals,
            "interventions": interventions
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)