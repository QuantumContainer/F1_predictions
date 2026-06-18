import pandas as pd

# Driver Statistics
def get_driver_stats(df, driver):
    """Returns:avg_finish,dnf_rate,avg_grid,avg_points"""

    driver_df = df[df["Driver"] == driver]

    # Handle rookies/new drivers
    if driver_df.empty:
        return {
            "avg_finish": 20,
            "dnf_rate": 0,
            "avg_grid": 20,
            "avg_points": 0}

    avg_finish = driver_df.loc[driver_df["DNF"] == 0,"FinishPosition"].mean()

    if pd.isna(avg_finish):
        avg_finish = 20

    return {
        "avg_finish": avg_finish,
        "dnf_rate": driver_df["DNF"].mean(),
        "avg_grid": driver_df["GridPosition"].mean(),
        "avg_points": driver_df["Points"].mean()}


# Circuit History
def get_circuit_history(df, driver, circuit):
    """    Returns:    circuit_avg_finish    circuit_dnf_rate    circuit_avg_points    circuit_races    """

    circuit_df = df[(df["Driver"] == driver) &(df["Race"] == circuit)]

    if circuit_df.empty:
        return {
            "circuit_avg_finish": 20,
            "circuit_dnf_rate": 0,
            "circuit_avg_points": 0,
            "circuit_races": 0}

    avg_finish = circuit_df.loc[circuit_df["DNF"] == 0,"FinishPosition"].mean()

    if pd.isna(avg_finish):
        avg_finish = 20

    return {
        "circuit_avg_finish": avg_finish,
        "circuit_dnf_rate": circuit_df["DNF"].mean(),
        "circuit_avg_points": circuit_df["Points"].mean(),
        "circuit_races": len(circuit_df)}

# Wet Weather Skill
def get_driver_wet_skill(df, driver):
    """    Returns:    wet_races    wet_avg_finish    """

    driver_df = df[df["Driver"] == driver]

    if driver_df.empty:
        return {"wet_races": 0,"wet_avg_finish": 20}

    wet_df = driver_df[driver_df["is_wet_race"] == 1]

    wet_races = len(wet_df)

    if wet_races == 0:
        return {"wet_races": 0,"wet_avg_finish": 20}

    wet_avg_finish = wet_df.loc[wet_df["DNF"] == 0,"FinishPosition"].mean()

    if pd.isna(wet_avg_finish):
        wet_avg_finish = 20

    return {
        "wet_races": wet_races,
        "wet_avg_finish": wet_avg_finish}

# Team / Car Performance

def get_car_performance(df, team):
    """    Returns:    team_avg_points    team_dnf_rate    team_avg_grid    team_podium_rate    """

    team_df = df[df["TeamName"] == team]

    if team_df.empty:
        return {
            "team_avg_points": 0,
            "team_dnf_rate": 0,
            "team_avg_grid": 20,
            "team_podium_rate": 0}

    num_races = team_df["Race"].nunique()

    team_avg_points = (team_df["Points"].sum() / num_races)

    team_dnf_rate = (team_df["DNF"].mean())

    team_avg_grid = (team_df["GridPosition"].mean())

    podiums = ((team_df["FinishPosition"] <= 3) &(team_df["DNF"] == 0)).sum()

    team_podium_rate = (podiums / len(team_df))

    return {
        "team_avg_points": team_avg_points,
        "team_dnf_rate": team_dnf_rate,
        "team_avg_grid": team_avg_grid,
        "team_podium_rate": team_podium_rate}


# Driver Summary Table
def get_all_driver_data(df):

    drivers = df["Driver"].unique()

    stats_list = []

    for driver in drivers:

        stats = get_driver_stats(df,driver)

        stats["Driver"] = driver

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)

    cols = ["Driver"] + [c for c in stats_df.columnsif c != "Driver"]

    stats_df = stats_df[cols]

    stats_df["avg_finish"] = (stats_df["avg_finish"].fillna(20))

    return stats_df


# Build Single Feature Row
def build_feature_row(historical_df,driver,team,circuit,current_grid=None):
    """
    Combines all feature functions into one row.
    Used for ML feature generation.
    """
    driver_stats = get_driver_stats(historical_df,driver)

    circuit_stats = get_circuit_history(historical_df,driver,circuit)

    wet_stats = get_driver_wet_skill(historical_df,driver)

    team_stats = get_car_performance(historical_df,team)
    recent3 = get_recent_form(historical_df,driver,3)

    recent5 = get_recent_form(historical_df,driver,5)
    feature_row = {"Driver": driver,"current_grid": current_grid,

        **driver_stats,
        **circuit_stats,
        **wet_stats,
        **team_stats,

        **recent3,
        **recent5}

    return feature_row

# Build Full Feature Matrix
def build_feature_matrix(historical_df,target_df):
    """
    historical_df = 2022 data
    target_df = 2023 data
    Creates training rows using
    2022 features and 2023 labels.
    """

    rows = []

    for _, row in target_df.iterrows():

        driver = row["Driver"]
        team = row["TeamName"]
        circuit = row["Race"]

        feature_row = build_feature_row(historical_df,driver,team,circuit,row["GridPosition"])

        feature_row["RaceName"] = row["Race"]
        feature_row["FinishPosition"] = row["FinishPosition"]

        rows.append(feature_row)

    feature_matrix = pd.DataFrame(rows)

    return feature_matrix

def get_recent_form(df, driver, n=5):

    driver_df = df[df["Driver"] == driver].sort_values(["Year", "Race"])

    if len(driver_df) == 0:
        return {
            f"last_{n}_avg_finish": 20,
            f"last_{n}_avg_points": 0}

    recent = driver_df.tail(n)

    avg_finish = recent[recent["DNF"] == 0]["FinishPosition"].mean()

    if pd.isna(avg_finish):
        avg_finish = 20

    return {
        f"last_{n}_avg_finish": avg_finish,
        f"last_{n}_avg_points":
            recent["Points"].mean() }

def build_prediction_matrix(historical_df,future_race_df):
    """
    future_race_df columns:

    Driver
    TeamName
    Race
    GridPosition
    """

    rows = []

    for _, row in future_race_df.iterrows():

        feature_row = build_feature_row(historical_df,row["Driver"],row["TeamName"],row["Race"],row["GridPosition"])

        feature_row["RaceName"] = row["Race"]

        rows.append(feature_row)

    return pd.DataFrame(rows)
