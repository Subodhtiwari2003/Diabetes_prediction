from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load(os.path.join('Diabetes_prediction', 'model.joblib'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract numeric inputs
        numeric_inputs = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]

        # Extract categorical input (BMI Category)
        bmi_category = request.form['bmi_category']

        # Combine numeric + categorical (as object)
        final_input = np.array(numeric_inputs + [bmi_category], dtype=object).reshape(1, -1)

        # Predict
        prediction = model.predict(final_input)
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    except Exception as e:
        result = f"Error during prediction: {str(e)}"

    return render_template('index.html', prediction_text=f'Prediction: {result}')


if __name__ == '__main__':
    app.run(debug=True)
