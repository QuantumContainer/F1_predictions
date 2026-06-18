import pandas as pd
import joblib

from data_loader import build_season_data
from feature_engineering import (
    build_feature_matrix,
    build_prediction_matrix
)
from model import train_model

# TRAIN FINAL MODEL
print("Loading historical data...")

season_2022 = build_season_data(2022)
season_2023 = build_season_data(2023)
season_2024 = build_season_data(2024)
season_2025 = build_season_data(2025)

history = pd.concat([season_2022,season_2023,season_2024,season_2025],ignore_index=True)

history = history[history["Race"] != "Pre-Season Testing"]

# CREATE TRAINING DATA
print("Building training matrix...")

feature_matrix = build_feature_matrix(history,history)

feature_matrix = feature_matrix.fillna(
    {
        "current_grid": 20,
        "avg_finish": 20,
        "avg_grid": 20,
        "avg_points": 0,
        "circuit_avg_finish": 20,
        "circuit_avg_points": 0,
        "wet_avg_finish": 20,
        "team_avg_points": 0,
        "team_avg_grid": 20,
        "team_podium_rate": 0,
        "last_3_avg_finish": 20,
        "last_3_avg_points": 0,
        "last_5_avg_finish": 20,
        "last_5_avg_points": 0,
        "FinishPosition": 20
    }
)

X = feature_matrix.drop(columns=["Driver","RaceName","FinishPosition"])

y = feature_matrix["FinishPosition"]

print("Training model...")

model = joblib.load("f1_model.pkl")

joblib.dump(model,"f1_model.pkl")

print("Model saved as f1_model.pkl")

# JAPANESE GP GRID
# Replace with actual qualifying positions

japanese_gp_grid = pd.DataFrame(
    [{"Driver":"RUS","TeamName":"Mercedes","Race":"Australian Grand Prix","GridPosition":1},
{"Driver":"ANT","TeamName":"Mercedes","Race":"Australian Grand Prix","GridPosition":2},
{"Driver":"HAD","TeamName":"Red Bull Racing","Race":"Australian Grand Prix","GridPosition":3},
{"Driver":"LEC","TeamName":"Ferrari","Race":"Australian Grand Prix","GridPosition":4},
{"Driver":"PIA","TeamName":"McLaren","Race":"Australian Grand Prix","GridPosition":5},
{"Driver":"NOR","TeamName":"McLaren","Race":"Australian Grand Prix","GridPosition":6},
{"Driver":"HAM","TeamName":"Ferrari","Race":"Australian Grand Prix","GridPosition":7},
{"Driver":"LAW","TeamName":"Racing Bulls","Race":"Australian Grand Prix","GridPosition":8},
{"Driver":"LIN","TeamName":"Racing Bulls","Race":"Australian Grand Prix","GridPosition":9},
{"Driver":"BOR","TeamName":"Audi","Race":"Australian Grand Prix","GridPosition":10},
{"Driver":"HUL","TeamName":"Audi","Race":"Australian Grand Prix","GridPosition":11},
{"Driver":"BEA","TeamName":"Haas F1 Team","Race":"Australian Grand Prix","GridPosition":12},
{"Driver":"OCO","TeamName":"Haas F1 Team","Race":"Australian Grand Prix","GridPosition":13},
{"Driver":"GAS","TeamName":"Alpine","Race":"Australian Grand Prix","GridPosition":14},
{"Driver":"ALB","TeamName":"Williams","Race":"Australian Grand Prix","GridPosition":15},
{"Driver":"COL","TeamName":"Alpine","Race":"Australian Grand Prix","GridPosition":16},
{"Driver":"ALO","TeamName":"Aston Martin","Race":"Australian Grand Prix","GridPosition":17},
{"Driver":"PER","TeamName":"Cadillac","Race":"Australian Grand Prix","GridPosition":18},
{"Driver":"STR","TeamName":"Aston Martin","Race":"Australian Grand Prix","GridPosition":19},
{"Driver":"BOT","TeamName":"Cadillac","Race":"Australian Grand Prix","GridPosition":20},
{"Driver":"SAI","TeamName":"Williams","Race":"Australian Grand Prix","GridPosition":21},
{"Driver":"VER","TeamName":"Red Bull Racing","Race":"Australian Grand Prix","GridPosition":22}]
)

# BUILD FUTURE FEATURES
future_matrix = build_prediction_matrix(history,japanese_gp_grid)

future_matrix = future_matrix.fillna(
    {
        "current_grid": 20,
        "avg_finish": 20,
        "avg_grid": 20,
        "avg_points": 0,
        "circuit_avg_finish": 20,
        "circuit_avg_points": 0,
        "wet_avg_finish": 20,
        "team_avg_points": 0,
        "team_avg_grid": 20,
        "team_podium_rate": 0,
        "last_3_avg_finish": 20,
        "last_3_avg_points": 0,
        "last_5_avg_finish": 20,
        "last_5_avg_points": 0
    }
)

X_future = future_matrix.drop(columns=["Driver","RaceName"])

future_matrix["PredictedFinish"] = (model.predict(X_future))

future_matrix = future_matrix.sort_values(by="PredictedFinish")

future_matrix["PredictedFinish"] = (future_matrix["PredictedFinish"].round(2))


# OUTPUT
print("\n")
print("=" * 60)
print("2026 Australian GP PREDICTION")
print("=" * 60)

print(future_matrix[["Driver","PredictedFinish"]])

print("\nPredicted Winner:")
print(future_matrix.iloc[0]["Driver"])

print("\nPredicted Podium:")
print(future_matrix.head(3)[["Driver","PredictedFinish"]])
