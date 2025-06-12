import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.data_processor import team_encodings, create_features, load_and_preprocess_data
from models.xgboost_model import fiting_model as fitting_xgb, predicting_outcomes as predicting_xgb, predicting_outcome_probability as predicting_probabilities_xgb

st.set_page_config(page_title="FIFA Prediction App", layout="wide")
st.title("FIFA World Cup Prediction App")

# Check if the data is already loaded
processed_df = load_and_preprocess_data()

@st.cache_data
def load_data():
    X, y = create_features(processed_df, return_target=True)
    return X, y

@st.cache_resource
def xgb_train(X, y):
    fitting_xgb(X, y)

X, y = load_data()
xgb_train(X, y)

team_encodings_dict = team_encodings()

# Creating a list of the home and away teams for the dropdowns
home_teams = list(team_encodings_dict['Home Team Name'].keys())
away_teams = list(team_encodings_dict['Away Team Name'].keys())

column1, column2 = st.columns(2)
with column1:
    home_team = st.selectbox("Home Team", home_teams)
with column2:
    away_team = st.selectbox("Away Team", away_teams)

if st.button("Predict Match Outcome"):
    match_stage = 'Final'
    input_data = pd.DataFrame({
        'Home Team Name': [team_encodings_dict['Home Team Name'][home_team]],
        'Away Team Name': [team_encodings_dict['Away Team Name'][away_team]],
        'Stage': [team_encodings_dict['Stage'].transform([match_stage])[0]] if 'Stage' in team_encodings_dict and team_encodings_dict['Stage'] is not None else [0],
        'Year': [2022],
        'Attendance': [0],
        'Half-time Home Goals': [0],
        'Half-time Away Goals': [0]
    })

    input_features = create_features(input_data)
    prediction = predicting_xgb(input_features)[0]

    outcome_probabilities = predicting_probabilities_xgb(input_features)[0]
    result_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    st.write(f"Predicted Outcome: {result_map[prediction]}")

    performance_chart = px.bar(
        pd.DataFrame({'Outcome': ['Away Win', 'Draw', 'Home Win'], 'Probability': outcome_probabilities}), x='Outcome', y='Probability', title='Prediction Probabilities', range_y=[0, 1])
    st.plotly_chart(performance_chart)

# The below chart and table are static. These values were from evaluating all three models against the same data
metrics = pd.DataFrame({
    'Accuracy': [0.45, 0.55, 0.52],
    'Precision': [0.44, 0.54, 0.51],
    'Recall': [0.43, 0.53, 0.50],
    'F1': [0.42, 0.52, 0.49]
}, index=['Mean', 'XGBoost', 'Neural Network'])

st.header("Model Comparison")
st.dataframe(metrics)

all_metrics = metrics.reset_index().melt(id_vars=['index'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1'], var_name='Metric', value_name='Score')

model_metric_chart = px.bar(all_metrics, x='index', y='Score', color='Metric', title='Model Performance Comparison', barmode='group', labels={'index': 'Models'}, range_y=[0, 0.8])
st.plotly_chart(model_metric_chart, config={"displayModeBar": False})
