from flask import Flask, request, jsonify
import joblib

filename = "model.pkl"
with open(filename, "rb") as f:
    model = joblib.load(f)

# Now you can use your model for predictions, like:
# prediction = model.predict([your_input_features])
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
