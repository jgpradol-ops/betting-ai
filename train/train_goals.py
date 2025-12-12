import os, joblib
from ml.goals_model import train_goals

X = [[1,2],[5,10],[8,4],[3,6]]
y_home = [2,1,0,3]
y_away = [1,0,2,1]

os.makedirs("models", exist_ok=True)

joblib.dump(train_goals(X, y_home), "models/goals_home.pkl")
joblib.dump(train_goals(X, y_away), "models/goals_away.pkl")

print("Modelos de goles entrenados correctamente")