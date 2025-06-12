import pandas as pd
from sklearn.model_selection import train_test_split
import models.mean_model as mean_model
import models.xgboost_model as xgboost_model
import models.neural_network_model as neural_network_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    evaluation_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return evaluation_metrics

def training_all_models(X_train, y_train):
    mean_model.fit(X_train, y_train)
    xgboost_model.fit(X_train, y_train)
    neural_network_model.fit(X_train, y_train)

# Evaluating each model against the test data
def evaluating_each_model(X_test, y_test):
    results = {}

    y_pred_mean = mean_model.predict(X_test)
    results['Mean'] = evaluate_model(y_test, y_pred_mean)

    y_pred_xgb = xgboost_model.predict(X_test)
    results['XGBoost'] = evaluate_model(y_test, y_pred_xgb)

    y_pred_nn = neural_network_model.predict(X_test)
    results['Neural Network'] = evaluate_model(y_test, y_pred_nn)
    return pd.DataFrame(results).T

# Calling all above functions to return the final training and performance metrics
def final_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    training_all_models(X_train, y_train)

    performance_metrics = evaluating_each_model(X_test, y_test)
    return performance_metrics 