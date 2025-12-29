from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, random_state):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model
