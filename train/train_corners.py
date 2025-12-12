import joblib, os
from sklearn.linear_model import LinearRegression

X = [[5,7],[3,4],[8,6]]
y = [12,7,14]

os.makedirs("models", exist_ok=True)
joblib.dump(LinearRegression().fit(X, y), "models/corners.pkl")