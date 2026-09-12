# Match Winner Prediction

A full-stack cricket match prediction project that predicts the likely winner of a T20 match using basic match factors such as team form, toss impact, venue, and date information.

## Screenshot

![Match Winner Predictor UI](pics/Screenshot%20from%202026-09-12%2016-48-24.png)

This screenshot shows the web interface where the user selects the two teams, toss winner, toss decision, venue, and match date, then gets a predicted winner with probability percentages.

---

## Project Overview

This project combines:

- Machine learning model training in Python
- A Go API to serve predictions
- An Astro frontend for user interaction
- Real CSV-based cricket match data for training and prediction

The system is designed to:

1. Load historical T20 match data
2. Preprocess and engineer meaningful features
3. Train a classifier model
4. Save the trained model to disk
5. Expose a prediction API
6. Show results in a simple web form

---

## Tech Stack

- Python
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - joblib
- Go
  - net/http
  - os/exec for calling the Python model
- Astro
  - frontend UI

---

## Project Structure

```text
Match-Winner-Prediction/
├── app/
│   ├── backend/
│   │   ├── go.mod
│   │   └── main.go
│   └── frontend/
│       ├── astro.config.mjs
│       ├── package.json
│       ├── tsconfig.json
│       └── src/
│           └── pages/
│               └── index.astro
├── ml_pipeline/
│   ├── data/
│   │   └── t20i_Matches_Data.csv
│   ├── model/
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   ├── match_winner_model.joblib
│   │   └── feature_columns.json
│   └── requirements.txt
├── pics/
│   └── Screenshot from 2026-09-12 16-48-24.png
├── .gitignore
├── README.md
└── .venv/
```

### File explanations

#### 1. ml_pipeline/model/train_model.py

This file handles the ML training flow.

It does the following:

- reads the cricket dataset
- cleans and filters invalid rows
- converts date values into usable features
- calculates recent team form
- creates target labels for match winner
- splits the dataset into training and test sets
- builds a preprocessing pipeline and XGBoost model
- saves the model and feature list for later prediction

#### 2. ml_pipeline/model/predict_model.py

This is the prediction script used by the backend.

It:

- reads the incoming request payload
- builds a one-row feature set from the input values
- loads the saved model
- predicts winner probability
- returns JSON output for the API

#### 3. app/backend/main.go

This is the Go backend.

It exposes:

- GET /health for health checks
- POST /predict for prediction requests

It receives a match payload, calls the Python prediction script, and returns a JSON response like:

```json
{
  "winner": "India",
  "team1_probability": 0.53,
  "team2_probability": 0.47,
  "message": "India is likely to win this match."
}
```

#### 4. app/frontend/src/pages/index.astro

This is the user-facing webpage.

It contains:

- team selection inputs
- venue field
- toss winner selection
- toss decision dropdown
- match date picker
- prediction button
- result panel with winner and percentage values

#### 5. ml_pipeline/data/t20i_Matches_Data.csv

This is the dataset used to train the model. It contains historical cricket match records used to predict future outcomes.

#### 6. pics/

This folder stores project screenshots for documentation and demo purposes.

---

## How the Model Works

The model uses a combination of engineered inputs such as:

- team 1 and team 2 names
- venue
- toss decision
- toss winner
- team recent form
- match month
- weekend flag

The target label is based on whether Team 1 won the match.

A preprocessing pipeline is used to handle categorical fields with OneHotEncoder and numeric features with StandardScaler before passing data into an XGBoost classifier.

---

## Setup Instructions

### 1. Clone the project

```bash
git clone <repository-url>
cd Match-Winner-Prediction
```

### 2. Create and use the project Python environment

The project already contains a virtual environment, or you can recreate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r ml_pipeline/requirements.txt
```

### 4. Train the model

```bash
python ml_pipeline/model/train_model.py
```

This creates:

- ml_pipeline/model/match_winner_model.joblib
- ml_pipeline/model/feature_columns.json

### 5. Run the backend

```bash
cd app/backend
go run .
```

The API will start on:

```text
http://localhost:8080
```

### 6. Run the frontend

```bash
cd app/frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:4321
```

---

## API Usage

### Health check

```bash
curl http://localhost:8080/health
```

### Prediction request

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "team1": "India",
    "team2": "Sri Lanka",
    "venue": "R Premadasa Stadium",
    "toss_winner": "India",
    "toss_decision": "bat",
    "match_date": "2026-09-12"
  }'
```

---

## Notes

- The model is a practical learning project and is meant for demonstration and experimentation.
- It can be improved with more features like team batting average, bowling strength, head-to-head stats, and recent match performance.
- The frontend is intentionally simple so it is easy to understand and extend.

---

## Summary

This project demonstrates a complete machine learning application pipeline:

- data preparation
- model training
- label generation
- model saving
- API integration
- user interface for prediction

It is a good example of how a machine learning model can be served in a real application environment.
