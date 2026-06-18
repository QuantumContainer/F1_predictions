import pandas as pd
import joblib

from data_loader import build_season_data
from feature_engineering import build_feature_matrix
from model import train_model

print("Loading seasons...")

season_2022 = build_season_data(2022)
season_2023 = build_season_data(2023)
season_2024 = build_season_data(2024)
season_2025 = build_season_data(2025)

for df in [season_2022, season_2023, season_2024, season_2025]:
    df.drop(df[df["Race"] == "Pre-Season Testing"].index,inplace=True)

print("Creating training examples...")

train_23 = build_feature_matrix(season_2022,season_2023)

train_24 = build_feature_matrix(
    pd.concat([season_2022, season_2023],ignore_index=True),season_2024)

train_25 = build_feature_matrix(pd.concat([season_2022,season_2023,season_2024],ignore_index=True),season_2025)

training_data = pd.concat([train_23,train_24,train_25],ignore_index=True)

training_data = training_data.fillna(20)

X = training_data.drop(columns=["Driver","RaceName","FinishPosition"])

y = training_data["FinishPosition"]

print("Training final model...")

model = train_model(X, y)

joblib.dump(model,"f1_model.pkl")

print("Saved model to f1_model.pkl")
