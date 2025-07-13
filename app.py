from flask import Flask, render_template, request
from joblib import load
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load(os.path.join('Diabetes_prediction', 'model.joblib'))

@app.route('/')
def home():
    return render_template('index.html')  # You’ll create this next

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from form
    features = [float(x) for x in request.form.values()]
    input_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == '__main__':
    app.run(debug=True)
