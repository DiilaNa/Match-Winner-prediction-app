import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 't20i_Matches_Data.csv'
MODEL_PATH = BASE_DIR / 'match_winner_model.joblib'
FEATURES_PATH = BASE_DIR / 'feature_columns.json'


def calculate_team_form(data, window=5):
    team_history = {}
    team1_form, team2_form = [], []

    for _, row in data.iterrows():
        t1, t2, w = row['team1'], row['team2'], row['winner']

        h1 = team_history.get(t1, [])
        team1_form.append(np.mean(h1[-window:]) if len(h1) > 0 else 0.5)

        h2 = team_history.get(t2, [])
        team2_form.append(np.mean(h2[-window:]) if len(h2) > 0 else 0.5)

        team_history.setdefault(t1, []).append(1 if w == t1 else 0)
        team_history.setdefault(t2, []).append(1 if w == t2 else 0)

    data['team1_form'] = team1_form
    data['team2_form'] = team2_form
    return data


def prepare_dataset(df):
    selected_columns = [
        'Match Date',
        'Team1 Name',
        'Team2 Name',
        'Match Venue (Stadium)',
        'Toss Winner',
        'Toss Winner Choice',
        'Match Winner',
    ]

    df = df[selected_columns].copy()
    df = df.rename(columns={
        'Match Date': 'date',
        'Team1 Name': 'team1',
        'Team2 Name': 'team2',
        'Match Venue (Stadium)': 'venue',
        'Toss Winner': 'toss_winner',
        'Toss Winner Choice': 'toss_decision',
        'Match Winner': 'winner',
    })

    df = df.dropna().reset_index(drop=True)
    df = df[df['winner'].isin(df['team1']) | df['winner'].isin(df['team2'])].reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'])
    df['match_month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df = df.sort_values('date').reset_index(drop=True)
    df = calculate_team_form(df)

    df['toss_winner_is_team1'] = (df['toss_winner'] == df['team1']).astype(int)
    df['team1_batting_first'] = np.where(
        (df['toss_winner_is_team1'] == 1) & (df['toss_decision'] == 'bat'), 1,
        np.where((df['toss_winner_is_team1'] == 0) & (df['toss_decision'] == 'field'), 1, 0)
    )
    df['target'] = (df['winner'] == df['team1']).astype(int)

    feature_cols = [
        'team1',
        'team2',
        'venue',
        'toss_decision',
        'team1_form',
        'team2_form',
        'toss_winner_is_team1',
        'team1_batting_first',
        'match_month',
        'is_weekend',
    ]

    X = df[feature_cols]
    y = df['target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def build_pipeline():
    categorical_features = ['team1', 'team2', 'venue', 'toss_decision']
    numerical_features = ['team1_form', 'team2_form', 'match_month']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features),
        ],
        remainder='passthrough',
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            objective='binary:logistic',
            eval_metric='logloss',
        )),
    ])
    return model


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_cols = prepare_dataset(df)

    model_pipeline = build_pipeline()
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(model_pipeline, MODEL_PATH)
    with FEATURES_PATH.open('w', encoding='utf-8') as f:
        json.dump(feature_cols, f)

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Features saved to: {FEATURES_PATH}')


if __name__ == '__main__':
    main()
