import joblib, os
from sklearn.linear_model import LogisticRegression

X = [[2,1],[0,0],[3,2],[1,1]]
y = [1,0,1,1]

os.makedirs("models", exist_ok=True)
joblib.dump(LogisticRegression().fit(X, y), "models/btts.pkl")