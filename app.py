from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load(r"booking_risk_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        lead_time = float(request.form["lead_time"])
        adr = float(request.form["adr"])
        total_nights = float(request.form["total_nights"])
        is_repeat = int(request.form["is_repeat"])
        special_requests = int(request.form["special_requests"])

        # Prepare input for model
        features = np.array([[lead_time, adr, total_nights, is_repeat, special_requests]])

        # Prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = "🔴 Risky Booking" if prediction == 1 else "🟢 Low Risk Booking"

        return render_template(
            "index.html",
            prediction=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
