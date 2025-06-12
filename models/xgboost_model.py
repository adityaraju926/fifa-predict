import xgboost as xgb
from sklearn.preprocessing import StandardScaler

model = None
scaler = None

def fiting_model(X, y):
    global model, scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y)

def predicting_outcomes(X):
    scaling = model.predict(scaler.transform(X))
    return scaling

def predicting_outcome_probability(X):
    outcome_probability = model.predict_proba(scaler.transform(X))
    return outcome_probability
        
