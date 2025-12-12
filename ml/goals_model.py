from sklearn.linear_model import PoissonRegressor

def train_goals(X, y):
    model = PoissonRegressor()
    model.fit(X, y)
    return model