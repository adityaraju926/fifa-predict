import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data.raw.world_cup_2022_teams import ALL_TEAMS

label_encoders = {}
all_matches_df = None

def load_and_preprocess_data():
    global all_matches_df, label_encoders
    
    all_matches_df = pd.read_csv("data/raw/WorldCupMatches.csv")

    # Adding only common teams between historical data and the 2022 map to the dataframe
    df = all_matches_df[all_matches_df['Home Team Name'].isin(ALL_TEAMS) & all_matches_df['Away Team Name'].isin(ALL_TEAMS)].copy()

    df['Datetime'] = pd.to_datetime(df['Datetime'], format='mixed')

    for feature in ['Year', 'Month', 'Day', 'DayOfWeek']:
        df[feature] = getattr(df['Datetime'].dt, feature.lower())

    # Substituting missing values with 0 for numerical columns
    numeric_cols = ['Home Team Goals', 'Away Team Goals', 'Attendance', 'Half-time Home Goals', 'Half-time Away Goals']
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Mapping the result to values based on the team that won
    df['Result'] = np.where(df['Home Team Goals'] > df['Away Team Goals'], 2, np.where(df['Home Team Goals'] < df['Away Team Goals'], 0, 1))
    
    for col in ['Home Team Name', 'Away Team Name', 'Stage']:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    all_matches_df = df

    return df

def team_encodings():
    user_input_encodings = {'Home Team Name': dict(zip(label_encoders['Home Team Name'].classes_, label_encoders['Home Team Name'].transform(label_encoders['Home Team Name'].classes_))), 'Away Team Name': dict(zip(label_encoders['Away Team Name'].classes_, label_encoders['Away Team Name'].transform(label_encoders['Away Team Name'].classes_))),'Stage': label_encoders['Stage'] if 'Stage' in label_encoders else None}

    return user_input_encodings

# Creating new feature columns for the training data
def create_features(df, return_target=False):
    global all_matches_df
    if all_matches_df is None:
        load_and_preprocess_data()
    
    team_stats = all_matches_df.groupby('Home Team Name').agg({'Home Team Goals': ['mean', 'sum'],'Result': ['mean', 'count']}).reset_index()

    team_stats.columns = ['Team', 'Avg_Goals_Scored', 'Total_Goals_Scored', 'Win_Rate', 'Matches_Played']

    df = df.merge(team_stats, left_on='Home Team Name', right_on='Team', how='left', suffixes=('', '_Home'))
    df = df.merge(team_stats, left_on='Away Team Name', right_on='Team', how='left', suffixes=('', '_Away'))
    df = df.drop(['Team', 'Team_Home', 'Team_Away'], axis=1, errors='ignore')

    feature_columns = ['Avg_Goals_Scored', 'Total_Goals_Scored', 'Win_Rate', 'Matches_Played', 'Avg_Goals_Scored_Away', 'Total_Goals_Scored_Away', 'Win_Rate_Away', 'Matches_Played_Away']
    
    if return_target:
        return df[feature_columns], df['Result']
    return df[feature_columns] 