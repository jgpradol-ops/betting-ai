from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# MODELOS GOLES
home_goals_model = joblib.load(os.path.join(MODELS_DIR, "goals_home.pkl"))
away_goals_model = joblib.load(os.path.join(MODELS_DIR, "goals_away.pkl"))

@app.route("/predict_goals", methods=["POST"])
def predict_goals():
    data = request.json
    X = [[
        data.get("home_rank", 10),
        data.get("away_rank", 10)
    ]]

    return jsonify({
        "predicted_home_goals": float(home_goals_model.predict(X)[0]),
        "predicted_away_goals": float(away_goals_model.predict(X)[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)