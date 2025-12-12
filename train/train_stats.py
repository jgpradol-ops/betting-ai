import joblib, os
from sklearn.linear_model import LinearRegression

X = [[10,5,3],[8,4,2],[12,6,5]]
y = [20,15,25]

os.makedirs("models", exist_ok=True)
joblib.dump(LinearRegression().fit(X, y), "models/stats.pkl")