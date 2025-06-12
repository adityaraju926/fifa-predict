import numpy as np

mean_probs = None

def fitting_model_mean(X, y):
    global mean_probs
    mean_probs = np.bincount(y) / len(y)

def predict_outcomes(X):
    average_probabilities = np.full(len(X), np.argmax(mean_probs))
    return average_probabilities

def predicting_probabilities(X):
    outcome_probabilities = np.tile(mean_probs, (len(X), 1)) 
    return outcome_probabilities