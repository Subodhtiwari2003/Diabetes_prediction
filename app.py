from flask import import flask, requests, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
