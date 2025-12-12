import joblib, os
from sklearn.linear_model import LogisticRegression

X = [[5,3],[2,1],[6,4]]
y = [1,0,1]

os.makedirs("models", exist_ok=True)
joblib.dump(LogisticRegression().fit(X, y), "models/scorer.pkl")