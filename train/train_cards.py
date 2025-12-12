import joblib, os
from sklearn.linear_model import LinearRegression

X = [[10,12],[8,7],[15,10]]
y = [5,3,7]

os.makedirs("models", exist_ok=True)
joblib.dump(LinearRegression().fit(X, y), "models/cards.pkl")