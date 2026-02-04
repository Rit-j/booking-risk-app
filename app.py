from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("booking_risk_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ---- Read inputs ----
        lead_time = float(request.form["lead_time"])
        adr = float(request.form["adr"])
        total_nights = float(request.form["total_nights"])
        is_repeat = int(request.form["is_repeat"])
        special_requests = int(request.form["special_requests"])

        # ---- Basic validation ----
        if lead_time < 0 or adr <= 0 or total_nights <= 0:
            return render_template(
                "index.html",
                error="Please enter valid, positive input values."
            )

        # ---- Model input ----
        features = np.array([[
            lead_time,
            adr,
            total_nights,
            is_repeat,
            special_requests
        ]])

        # ---- Prediction ----
        probability = model.predict_proba(features)[0][1] * 100

        # ---- TWO-CLASS LOGIC ONLY ----
        if probability >= 50:
            prediction = "High Risk Booking"
            risk_class = "risk"
            recommendation = (
                "High cancellation risk detected. "
                "Consider flexible cancellation policies or advance payment."
            )
        else:
            prediction = "Low Risk Booking"
            risk_class = "safe"
            recommendation = (
                "Low cancellation risk. "
                "Good opportunity to upsell add-ons or premium services."
            )

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability, 2),
            risk_class=risk_class,
            recommendation=recommendation
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=f"Input processing error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
