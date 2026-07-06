from flask import Flask, render_template, request,url_for
import pandas as pd
import numpy as np
import joblib
import os
import csv
from datetime import datetime

app = Flask(__name__)

model = joblib.load(r"C:\Users\abc\OneDrive\Desktop\anand collage 45 days traning\project\Telco-Customer-Churn\model\RandomForestClassifier.lb")
HISTORY_FILE = "history.csv"

if not os.path.exists(HISTORY_FILE):

    with open(HISTORY_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Date",
            "Prediction",
            "Probability"
        ])
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/project")
def project():
    return render_template("project.html")

@app.route("/history")
def history():

    if os.path.exists(HISTORY_FILE):

        history = pd.read_csv(HISTORY_FILE)

        history = history.to_dict(orient="records")

    else:

        history = []

    return render_template(
        "history.html",
        history=history
    )
 
@app.route("/predict", methods=["POST"])
def predict():

    try:

        gender = int(request.form["gender"])
        SeniorCitizen = int(request.form["SeniorCitizen"])
        Partner = int(request.form["Partner"])
        Dependents = int(request.form["Dependents"])
        tenure = int(request.form["tenure"])
        PhoneService = int(request.form["PhoneService"])
        MultipleLines = int(request.form["MultipleLines"])
        InternetService = int(request.form["InternetService"])
        OnlineSecurity = int(request.form["OnlineSecurity"])
        OnlineBackup = int(request.form["OnlineBackup"])
        DeviceProtection = int(request.form["DeviceProtection"])
        TechSupport = int(request.form["TechSupport"])
        StreamingTV = int(request.form["StreamingTV"])
        StreamingMovies = int(request.form["StreamingMovies"])
        Contract = int(request.form["Contract"])
        PaperlessBilling = int(request.form["PaperlessBilling"])
        PaymentMethod = int(request.form["PaymentMethod"])
        MonthlyCharges = float(request.form["MonthlyCharges"])
        TotalCharges = float(request.form["TotalCharges"])

        data = pd.DataFrame([[
            gender,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            InternetService,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            Contract,
            PaperlessBilling,
            PaymentMethod,
            MonthlyCharges,
            TotalCharges
        ]],
                            columns=[

            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"

        ])

        prediction = model.predict(data)[0]

        probability = model.predict_proba(data)

        confidence = round(
            np.max(probability) * 100,
            2
        )

        if prediction == 1:

            result = "Customer Will Churn"

        else:

            result = "Customer Will Stay"
         
        with open(HISTORY_FILE, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                result,
                confidence
            ])

        return render_template(
            "project.html",
            prediction=result,
            probability=confidence
        )

    except Exception as e:

        return render_template(
            "project.html",
            prediction=f"Error : {str(e)}"
        )
if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )