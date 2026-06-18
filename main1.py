from sklearn.metrics import mean_absolute_error
from data_loader import build_season_data
from feature_engineering import build_feature_matrix
from model import train_model
import pandas as pd

# Evaluation Function
def evaluate_predictions(df):

    total_races = 0
    top1_correct = 0
    top3_correct = 0

    for race_name, race_df in df.groupby("RaceName"):

        total_races += 1

        race_df = race_df.sort_values(by="PredictedPosition",ascending=True)

        predicted_winner = race_df.iloc[0]["Driver"]

        actual_winner = race_df[race_df["FinishPosition"] == 1].iloc[0]["Driver"]

        if predicted_winner == actual_winner:
            top1_correct += 1

        predicted_top3 = race_df.head(3)["Driver"].tolist()

        if actual_winner in predicted_top3:
            top3_correct += 1

    return {
        "top1_accuracy": round(100 * top1_correct / total_races,2),
        "top3_accuracy": round(100 * top3_correct / total_races,2),
        "total_races": total_races	}

# Load Seasons
history_2022 = build_season_data(2022)

season_2023 = build_season_data(2023)

# Remove pre-season testing
season_2023 = season_2023[season_2023["Race"] != "Pre-Season Testing"]

# Build Feature Matrix
feature_matrix = build_feature_matrix(history_2022,season_2023)

feature_matrix = feature_matrix.fillna(
    {
        "avg_finish": 20,
        "avg_grid": 20,
        "avg_points": 0,
        "circuit_avg_finish": 20,
        "circuit_avg_points": 0,
        "wet_avg_finish": 20,
        "team_avg_points": 0,
        "team_avg_grid": 20,
        "team_podium_rate": 0,
        "FinishPosition": 20,
        "last_3_avg_finish": 20,
        "last_3_avg_points": 0,
        "last_5_avg_finish": 20,
        "last_5_avg_points": 0,
        "current_grid": 20
    }
)

print("\nFeature Matrix Shape:")
print(feature_matrix.shape)

# Train Model
X = feature_matrix.drop(
    columns=["Driver","RaceName","FinishPosition"])

y = feature_matrix["FinishPosition"]

model = train_model(X, y)

# Predict
predictions = model.predict(X)

feature_matrix["PredictedPosition"] = predictions

# MAE
mae = mean_absolute_error(y,predictions)

print(f"\nMAE: {mae:.3f}")


# Prediction Table
results_table = feature_matrix[["RaceName","Driver","FinishPosition","PredictedPosition"]].copy()

results_table["PredictedPosition"] = (results_table["PredictedPosition"].round(2))

results_table = results_table.sort_values(by="PredictedPosition")

print("\nTop Predictions:\n")
print(results_table.head(30))


# Race-Level Accuracy
metrics = evaluate_predictions(feature_matrix)

print("\nRace Prediction Metrics:")
print(f"Top-1 Accuracy: " f"{metrics['top1_accuracy']}%")

print(f"Top-3 Accuracy: " f"{metrics['top3_accuracy']}%")

print(f"Races Evaluated: " f"{metrics['total_races']}")

# Feature Importance
importance_df = pd.DataFrame({"Feature": X.columns,"Importance": model.feature_importances_})

importance_df = importance_df.sort_values(by="Importance",ascending=False)

print("\nTop Features:\n")
print(importance_df.head(15))

