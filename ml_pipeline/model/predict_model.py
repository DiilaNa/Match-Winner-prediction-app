import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'match_winner_model.joblib'
FEATURES_PATH = BASE_DIR / 'feature_columns.json'
DATA_PATH = BASE_DIR.parent / 'data' / 't20i_Matches_Data.csv'


def get_team_form(team_name: str, history: dict[str, list[int]], window: int = 5) -> float:
    if team_name not in history or len(history[team_name]) == 0:
        return 0.5
    recent = history[team_name][-window:]
    return float(np.mean(recent))


def build_feature_row(match_request: dict) -> dict:
    team1 = match_request['team1']
    team2 = match_request['team2']
    venue = match_request['venue']
    toss_winner = match_request['toss_winner']
    toss_decision = str(match_request['toss_decision']).strip().lower()
    match_date = pd.to_datetime(match_request.get('match_date', pd.Timestamp.now().strftime('%Y-%m-%d')))

    df = pd.read_csv(DATA_PATH)
    df = df[['Match Date', 'Team1 Name', 'Team2 Name', 'Match Winner']].copy()
    df['Match Date'] = pd.to_datetime(df['Match Date'])
    df = df.sort_values('Match Date').reset_index(drop=True)

    history: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        if pd.to_datetime(row['Match Date']) >= match_date:
            continue

        team1_name = row['Team1 Name']
        team2_name = row['Team2 Name']
        winner = row['Match Winner']

        history.setdefault(team1_name, []).append(1 if winner == team1_name else 0)
        history.setdefault(team2_name, []).append(1 if winner == team2_name else 0)

    team1_form = get_team_form(team1, history)
    team2_form = get_team_form(team2, history)

    feature_row = {
        'team1': team1,
        'team2': team2,
        'venue': venue,
        'toss_decision': toss_decision,
        'team1_form': team1_form,
        'team2_form': team2_form,
        'toss_winner_is_team1': int(toss_winner == team1),
        'team1_batting_first': int((toss_winner == team1 and toss_decision == 'bat') or (toss_winner != team1 and toss_decision == 'field')),
        'match_month': int(match_date.month),
        'is_weekend': int(match_date.dayofweek in [5, 6]),
    }

    return feature_row


def main():
    request_json = sys.argv[1] if len(sys.argv) > 1 else '{}'
    match_request = json.loads(request_json)

    feature_row = build_feature_row(match_request)

    feature_columns = json.loads(FEATURES_PATH.read_text(encoding='utf-8'))
    df_input = pd.DataFrame([feature_row], columns=feature_columns)

    model = joblib.load(MODEL_PATH)
    probability = model.predict_proba(df_input)[0]
    prediction = model.predict(df_input)[0]

    team1_probability = float(probability[1])
    team2_probability = float(probability[0])
    winner = match_request['team1'] if prediction == 1 else match_request['team2']

    result = {
        'winner': winner,
        'team1_probability': team1_probability,
        'team2_probability': team2_probability,
        'message': f'{winner} is likely to win this match.',
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
