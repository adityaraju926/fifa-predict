import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

model = None
scaler = None

def creating_network(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, 3))

def fitting_model(X, y):
    global model, scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Converting to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)
    
    model = creating_network(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for _ in range(2): 
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

def predicting_outcomes(X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(scaler.transform(X))
        outputs = model(X_tensor)
        return outputs.argmax(dim=1).numpy()

def predicting_probabilities(X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(scaler.transform(X))
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        return probs.numpy() 