
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    funded_amount = float(request.form['funded_amount'])
    clean_borrower_genders = int(request.form['clean_borrower_genders'])
    term_in_months = int(request.form['term_in_months'])
    lender_count = int(request.form['lender_count'])
    loan_amount_log = float(request.form['loan_amount_log'])
    sector = int(request.form['sector'])
    activity = int(request.form['activity'])
    region = int(request.form['region'])
    country_code = int(request.form['country_code'])

    # Prepare the input array
    input_features = np.array([[funded_amount, clean_borrower_genders, term_in_months, lender_count, 
                                loan_amount_log, sector, activity, region, country_code]])

    # Make a prediction
    prediction = model.predict(input_features)
    prediction = prediction[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
