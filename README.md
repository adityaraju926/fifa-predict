# FIFA World Cup Prediction App

## Problem
This project predicts the outcomes of FIFA World Cup matches since the 2026 tournament is around the corner. The goal is to build and evaluate different models that can accurately predict whether a match will result in a home win, draw, or away win based on historical data and team stats.

## Data Sources
- **WorldCupMatches.csv**: Contains all historical match data, including team names, goals scored, and match stages.
- **WorldCupPlayers.csv**: Contains player data for each match.
- **WorldCups.csv**: Contains information about past World Cup tournaments only.

## Literature Review
There have been previous attempts of creating a FIFA Prediction App, but this one differs significantly. This project builds upon these existing approaches by incorporating team metrics and advanced models to improve accuracy. While similar projects exist such as ([FIFA Player Ratings Prediction](https://github.com/awwalm/FIFAPrediction), [World Cup Predictor](https://github.com/neaorin/PredictTheWorldCup), [FIFA 23 Predictor](https://github.com/Sahil-R-Kale/fifa-23-predictor)), this implementation offers several improvements including but not limited to:

- Multiple model performance metrics are shown for comparison
- The UI is minimal which allows the user to select and quickly get the predicted outcome
- In-depth preprocessing, label encoding, and feature engineering is used for the consumption of the model

## Model Evaluation Process & Metric Selection
The models are evaluated using:
- **Accuracy**: Measures validity of predictions against true values. Selected this metric because it helps measure the overall "correctness."
- **Precision**: Measures how reliable the positive predictions. Selected this metric since it highlights the "true positives."
- **Recall**: Measures if the models are making false negative predictions. Selected this metric because it helps ensure that potential outcomes aren't missed.
- **F1 Score**: Combination of precision and recall. Selected this metric to help identify models that are good at both and accounts for any trade-off between the two.

## Modeling Approach
Initially the following models were evaluated against the same data:
1. Mean Model
2. XGBoost
3. Neural Network

Upon review of the performance metrics, XGBoost had the best values so the predicted outcomes are through this model.

## Data Processing Pipeline
1. **Data Loading**: Load data from the raw data files
2. **Preprocessing**: Filter data for relevant features, missing values, and encode for model training
3. **Feature Engineering**: Create new features based on existing data
4. **Model Training**: Training all models using the processed data
5. **Model Evaluation**: Evaluate all models with the same metrics for comparison


## Comparison to Naive Approach
The XGBoost and Neural Network models are compared to the Naive Model to show an improvement in prediction accuracy. In this use case, both models being compared to the Naive Model has significantly better performance.

## Ethics Statement
This project only utilizes datasets that are publically available and the models are transparent with no biases towards any team or outcome.

## Directory Structure

- `main.py` — UI
- `models/mean_model.py` — Naive (mean) model code
- `models/xgboost_model.py` — XGBoost model code
- `models/neural_network_model.py` — Neural network code
- `scripts/data_processor.py` — Data loading, preprocessing, and feature engineering
- `data/` — Contains the raw data files and 2022 team map

## How to Run the App

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the app:
   ```sh
   streamlit run main.py
   ```
3. Open the local URL provided by Streamlit in your browser.

## Why XGBoost Performs The Best & Future Improvements

XGBoost outperforms both the naive and neural network models in this project due to it being able to capture non-linear relationships, even with limited samples and features. The naive model relies only on historical outcomes without considering features, while the neural network may need more data, better tuning, or enhanced feature to perform better than XGBoost. XGBoost can be improved with hyperparameter tuning, adding more feaetures, or even ensemble methods.

## Project Structure
```
fifa-predict/
├── data/
│   ├── raw/                     
├── models/                      
│   ├── mean_model.py   
│   ├── xgboost_model.py 
│   ├── neural_network_model.py 
│   └── train.py         
├── scripts/
│   └── data_processor.py 
├── main.py              
├── requirements.txt     
└── README.md           
```