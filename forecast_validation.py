import pandas as pd

from data_loader import build_season_data
from backtest import run_backtest
from main1 import X, model

# Load Seasons
season_2022 = build_season_data(2022)
season_2023 = build_season_data(2023)
season_2024 = build_season_data(2024)
season_2025 = build_season_data(2025)

# Backtest 1
# 2022 -> Predict 2023
print("\n")
print("=" * 60)
print("2022 -> PREDICT 2023")
print("=" * 60)

model23, results23, mae23, metrics23 = (run_backtest(season_2022,season_2023))

# Backtest 2
# 2022+2023 -> Predict 2024
history_22_23 = pd.concat([season_2022,season_2023],ignore_index=True)

print("\n")
print("=" * 60)
print("2022+2023 -> PREDICT 2024")
print("=" * 60)

model24, results24, mae24, metrics24 = (run_backtest(history_22_23,season_2024))

# Backtest 3
# 2022+2023+2024 -> Predict 2025
history_22_24 = pd.concat([season_2022,season_2023,season_2024],ignore_index=True)

print("\n")
print("=" * 60)
print("2022+2023+2024 -> PREDICT 2025")
print("=" * 60)

model25, results25, mae25, metrics25 = (run_backtest(history_22_24,season_2025))

# Summary Table
summary = pd.DataFrame([{
            "Forecast":"2022 -> 2023",

            "MAE":round(mae23, 3),

            "Top1":metrics23["top1_accuracy"],

            "Top3":metrics23["top3_accuracy"]    },

        {    "Forecast":"2022+2023 -> 2024",

            "MAE":round(mae24, 3),

            "Top1":metrics24["top1_accuracy"],

            "Top3":metrics24["top3_accuracy"]    },

        {    "Forecast":"2022+2023+2024 -> 2025",

            "MAE":round(mae25, 3),

            "Top1":metrics25["top1_accuracy"],

            "Top3":metrics25["top3_accuracy"]    }    ]    )

print("\n")
print("=" * 60)
print("FORECAST SUMMARY")
print("=" * 60)

print(summary)

importance_df = pd.DataFrame({"Feature": X.columns,"Importance": model.feature_importances_})

print(importance_df.sort_values(by="Importance",ascending=False).head(20))
