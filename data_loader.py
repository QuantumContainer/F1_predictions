import os
import fastf1
import pandas as pd

# ============================================================
# FastF1 Cache
# ============================================================

fastf1.Cache.enable_cache("data/cache")


# ============================================================
# Weather Features
# ============================================================

def get_weather_features(session):

    weather = session.weather_data

    return {
        "avg_temp": weather["AirTemp"].mean(),
        "avg_rainfall": weather["Rainfall"].mean(),
        "is_wet_race": int(weather["Rainfall"].mean() > 0)
    }


# ============================================================
# Build Season Data
# ============================================================

def build_season_data(year, force_refresh=False):

    os.makedirs(
        "data/seasons",
        exist_ok=True
    )

    filepath = f"data/seasons/{year}_season.csv"

    # --------------------------------------------------------
    # Load Cached Version
    # --------------------------------------------------------

    if os.path.exists(filepath) and not force_refresh:

        print(
            f"Loading cached season data for {year}"
        )

        return pd.read_csv(filepath)

    # --------------------------------------------------------
    # Build Season From FastF1
    # --------------------------------------------------------

    print(
        f"Building season data for {year}"
    )

    schedule = fastf1.get_event_schedule(year)

    all_races = []

    for _, event in schedule.iterrows():

        race_name = event["EventName"]

        try:

            print(
                f"Loading {year} - {race_name}"
            )

            session = fastf1.get_session(
                year,
                race_name,
                "R"
            )

            session.load()

            race_df = session.results[
                [
                    "DriverNumber",
                    "Abbreviation",
                    "TeamName",
                    "GridPosition",
                    "Position",
                    "Points",
                    "Status",
                    "Laps"
                ]
            ].copy()

            race_df.rename(
                columns={
                    "Abbreviation": "Driver",
                    "Position": "FinishPosition"
                },
                inplace=True
            )

            weather = get_weather_features(
                session
            )

            race_df["Year"] = year
            race_df["Race"] = race_name

            race_df["avg_temp"] = (weather["avg_temp"])

            race_df["avg_rainfall"] = (weather["avg_rainfall"])

            race_df["is_wet_race"] = (weather["is_wet_race"])

            all_races.append(race_df)

        except Exception as e:
            print(f"Failed to load {race_name}: {e}")

    # --------------------------------------------------------
    # Combine All Races
    # --------------------------------------------------------

    season_df = pd.concat(all_races,ignore_index=True)

    season_df["DNF"] = (season_df["Status"] != "Finished").astype(int)

    # --------------------------------------------------------
    # Save To Disk
    # --------------------------------------------------------

    season_df.to_csv(filepath,index=False)

    print(f"Saved season data to {filepath}")

    return season_df