from sklearn.ensemble import RandomForestRegressor


def train_model(X, y):

    model = RandomForestRegressor(n_estimators=1000,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1)

    model.fit(X, y)

    return model
